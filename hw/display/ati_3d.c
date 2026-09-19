/*
 * QEMU ATI SVGA emulation - Rage 128 CCE (PM4) command ring decoder
 *
 * Decodes the packet0/1/2/3 command stream Rage 128 hardware calls the
 * "PM4" engine (ATI's driver-side documentation and the open source
 * Linux/XFree86 stack call the same unit the "CCE" - Concurrent Command
 * Engine). Packet layout and opcodes below are taken from the public
 * r128_reg.h used by the XFree86/X.Org r128 driver and the Linux r128
 * DRM driver, not reverse engineered or guessed.
 *
 * Current scope:
 *   - Packet0 (register write burst) and Packet1 (dual register write):
 *     fully implemented, dispatched through the same ati_mm_write() path
 *     MMIO writes use, so any guest behaviour that depends on register
 *     side effects (e.g. latching dst_offset, triggering a 2D blt via
 *     DP_GUI_MASTER_CNTL, etc.) works whether the guest talks to the
 *     card via raw MMIO or via the CCE ring.
 *   - Packet2 (nop/pad): implemented (no-op).
 *   - Packet3: opcode is decoded and dispatched, but only far enough to
 *     keep the ring pointer in sync (i.e. we always consume the right
 *     number of payload dwords). The 2D packet3 ops (PAINT, BITBLT, ...)
 *     are not yet translated into calls into ati_2d_blt(), and the 3D
 *     draw opcodes (3D_RNDR_GEN_INDX_PRIM / 3D_RNDR_GEN_PRIM) are
 *     decoded (primitive type, vertex format, vertex count) and logged,
 *     but no rasterization happens yet - that is the next milestone.
 *   - Only the pure-PIO ring mode (PM4_BUFFER_CNTL_192PIO) is handled.
 *     The bus-mastering/indirect-buffer modes are logged once and
 *     otherwise ignored, since they require DMA reads from guest system
 *     memory rather than the card's own VRAM.
 *
 * Copyright (c) 2026
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 */

#include "qemu/osdep.h"
#include "ati_int.h"
#include "ati_regs.h"
#include "qemu/log.h"
#include "qemu/bswap.h"
#include <math.h>
#include "hw/pci/pci_device.h"

/*
 * Safety cap on how many ring dwords we will walk in one call. The ring
 * itself is only PM4_192PIO_RING_DWORDS deep, but a guest could in
 * theory keep advancing wptr faster than we drain, or a decode bug could
 * desync us and spin - bound total work per call rather than trust
 * rptr/wptr to always be sane.
 */
/*
 * Safety cap on how many ring dwords we will walk in one call, bounding
 * work per call rather than trusting rptr/wptr to always be sane.
 */
#define CCE_MAX_DWORDS_PER_CALL (PM4_192PIO_RING_DWORDS * 4)

/*
 * Separate, much larger cap for a single indirect buffer.
 *
 * This used to share CCE_MAX_DWORDS_PER_CALL (768), a figure derived from
 * the 192-dword PIO ring - which has nothing to do with how big a command
 * buffer in system memory may be. PM4_IW_INDSIZE is a 24-bit field, and
 * the guest routinely submits buffers of 0x1470 (5232) dwords; clamping
 * those to 768 threw away ~86% of each one, so most of every frame's
 * geometry never reached the rasterizer at all. Sized to hold the buffers
 * actually observed with generous headroom, while still bounding the work
 * done for one submission.
 */
/*
 * Sanity limit on one indirect buffer, in dwords.
 *
 * Neither r128_reg.h nor the Radeon header documents the width of the
 * CP_IB_BUFSZ (0x073c) size field, so this is a plausibility bound rather
 * than a decoded field mask. Measured from a Tux Racer capture: of 9895
 * submissions, 9648 have nothing set above bit 13 and are almost all
 * between 2 and 96 dwords; the rest carried a size-class flag in the top
 * bit, which ati.c now strips before calling here.
 *
 * With that stripping in place the remaining large values - a band around
 * 21000-22000 dwords - are genuine buffers, and a 16384 bound was
 * discarding 1268 of them per run, which is real work the driver expects
 * to have executed. This is now only a backstop against a wild value, not
 * a plausibility filter.
 */
#define CCE_MAX_INDIRECT_DWORDS 262144

/* Largest packet3 payload we'll buffer locally before dispatching. */
#define CCE_MAX_PACKET3_PAYLOAD ATI_CCE_MAX_PKT3

static uint32_t ati_cce_ring_read(ATIVGAState *s, uint32_t dword_index)
{
    uint32_t ring_off = dword_index % PM4_192PIO_RING_DWORDS;
    hwaddr byte_off = s->cce.buffer_offset + ring_off * 4;

    if (byte_off > s->vga.vram_size - 4) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "ati_cce: ring read outside vram (off=0x%x)\n",
                      (unsigned)byte_off);
        return 0;
    }
    return ldl_le_p(s->vga.vram_ptr + byte_off);
}


/*
 * Minimal 3D rasteriser.
 *
 * The vertex format word and the VC_CNTL word arrive in the packet payload,
 * not in PM4_VC_FORMAT / PM4_VC_CNTL - the guest never writes those
 * registers, which is why every draw used to report all-zero state:
 *
 *     payload[0]  vertex format
 *     payload[1]  VC_CNTL: primitive type, walk mode, vertex count
 *     payload[2+] vertex data, for the inline (PRIM_WALK_RING) form
 *
 * Vertices are floats. The first three are screen-space x, y and z; any
 * remaining components up to four are taken as r, g, b, a in 0..1. The
 * stride is derived from the payload length and the vertex count rather
 * than decoded from the format word, which keeps this working across the
 * formats without a complete format table.
 *
 * Scope: solid and Gouraud-shaded triangles into the current destination
 * surface, scissor-clipped, 32bpp only. No depth buffering, no texturing,
 * no alpha blending yet - alpha is ignored rather than applied.
 */
static inline float ati_3d_f(uint32_t bits)
{
    union { uint32_t u; float f; } v = { .u = bits };
    return v.f;
}

/* Defined below, next to the indirect buffer fetch that also uses it. */
static bool ati_gart_translate(ATIVGAState *s, uint32_t off, hwaddr *pa);
static bool ati_tex_loc_lookup(ATIVGAState *s, uint32_t page, bool *in_gart);
static bool ati_addr_in_display_buffer(ATIVGAState *s, uint32_t off);
static void ati_tex_loc_record(ATIVGAState *s, uint32_t page, bool in_gart);

/*
 * Vertex format bits (r128_reg.h, R128_CCE_VC_FRMT_*). The position x, y, z
 * is always present; every other component is optional and appears in bit
 * order after it. Component sizes are in dwords.
 */
#define VC_FRMT_RHW          0x00000001
#define VC_FRMT_DIFFUSE_BGR  0x00000002
#define VC_FRMT_DIFFUSE_A    0x00000004
#define VC_FRMT_DIFFUSE_ARGB 0x00000008
#define VC_FRMT_SPEC_BGR     0x00000010
#define VC_FRMT_SPEC_F       0x00000020
#define VC_FRMT_SPEC_FRGB    0x00000040
#define VC_FRMT_S_T          0x00000080
#define VC_FRMT_S2_T2        0x00000100
#define VC_FRMT_RHW2         0x00000200

typedef struct ATI3DLayout {
    unsigned stride;     /* dwords per vertex */
    int off_bgr;         /* dword offset of diffuse b,g,r, -1 if absent */
    int off_a;           /* dword offset of diffuse alpha, -1 if absent */
    int off_argb;        /* packed ARGB dword, -1 if absent */
    int off_st;          /* dword offset of s,t, -1 if absent */
    int off_rhw;         /* dword offset of 1/w, -1 if absent */
    int off_s2t2;        /* dword offset of s2,t2, -1 if absent */
    bool st_direct;      /* SETUP_CNTL TEXTURE_ST_DIRECT: skip the RHW divide */
} ATI3DLayout;

static void ati_3d_layout(uint32_t fmt, ATI3DLayout *l)
{
    unsigned n = 3;      /* x, y, z */

    l->off_bgr = l->off_a = l->off_argb = l->off_st = -1;
    l->off_rhw = -1;
    l->off_s2t2 = -1;

    if (fmt & VC_FRMT_RHW) {
        l->off_rhw = n;
        n++;
    }
    if (fmt & VC_FRMT_DIFFUSE_BGR) {
        l->off_bgr = n;
        n += 3;
    }
    if (fmt & VC_FRMT_DIFFUSE_A) {
        l->off_a = n;
        n++;
    }
    if (fmt & VC_FRMT_DIFFUSE_ARGB) {
        l->off_argb = n;
        n++;
    }
    if (fmt & VC_FRMT_SPEC_BGR) {
        n += 3;
    }
    if (fmt & VC_FRMT_SPEC_F) {
        n++;
    }
    if (fmt & VC_FRMT_SPEC_FRGB) {
        n += 3;
    }
    if (fmt & VC_FRMT_S_T) {
        l->off_st = n;
        n += 2;
    }
    if (fmt & VC_FRMT_S2_T2) {
        l->off_s2t2 = n;
        n += 2;
    }
    if (fmt & VC_FRMT_RHW2) {
        n++;
    }
    l->stride = n;
}

typedef struct ATI3DVert {
    float x, y, z;
    float r, g, b, a;
    float s, t;
    float s2, t2;
    float rhw;
} ATI3DVert;

static void ati_3d_fetch_vert(const uint32_t *p, const ATI3DLayout *l,
                              ATI3DVert *v)
{
    v->x = ati_3d_f(p[0]);
    v->y = ati_3d_f(p[1]);
    v->z = ati_3d_f(p[2]);

    v->r = v->g = v->b = 1.0f;
    v->a = 1.0f;

    if (l->off_bgr >= 0) {
        /* Note the order: blue, green, red - the bit is DIFFUSE_BGR. */
        v->b = ati_3d_f(p[l->off_bgr + 0]);
        v->g = ati_3d_f(p[l->off_bgr + 1]);
        v->r = ati_3d_f(p[l->off_bgr + 2]);
    }
    if (l->off_a >= 0) {
        v->a = ati_3d_f(p[l->off_a]);
    }
    v->s = v->t = v->s2 = v->t2 = 0.0f;
    v->rhw = 1.0f;
    if (l->off_rhw >= 0) {
        v->rhw = ati_3d_f(p[l->off_rhw]);
    }
    if (l->off_st >= 0) {
        v->s = ati_3d_f(p[l->off_st]);
        v->t = ati_3d_f(p[l->off_st + 1]);
        /*
         * With RHW present the texture coordinates arrive premultiplied by
         * 1/w. Recover them so they can be interpolated in the units the
         * texture is actually indexed in; using the premultiplied values
         * directly made s,t barely vary across a surface, which sampled
         * close to a single texel and produced flat blocks of colour.
         */
        /*
         * SETUP_CNTL bit 9 (TEXTURE_ST_DIRECT vs _ST_MULT_W) tells the
         * hardware whether S,T arrive already final or need this divide.
         * A real cube-rendering session showed ST_DIRECT selected, meaning
         * this divide should NOT run - it was being applied unconditionally
         * whenever RHW was present in the vertex format, which is wrong
         * whenever the driver asks for direct coordinates instead.
         */
        if (!l->st_direct && l->off_rhw >= 0 && v->rhw != 0.0f) {
            v->s /= v->rhw;
            v->t /= v->rhw;
        }
    }
    if (l->off_s2t2 >= 0) {
        v->s2 = ati_3d_f(p[l->off_s2t2]);
        v->t2 = ati_3d_f(p[l->off_s2t2 + 1]);
        if (!l->st_direct && l->off_rhw >= 0 && v->rhw != 0.0f) {
            v->s2 /= v->rhw;
            v->t2 /= v->rhw;
        }
    }
    if (l->off_argb >= 0) {
        uint32_t c = p[l->off_argb];

        v->a = ((c >> 24) & 0xff) / 255.0f;
        v->r = ((c >> 16) & 0xff) / 255.0f;
        v->g = ((c >> 8) & 0xff) / 255.0f;
        v->b = (c & 0xff) / 255.0f;
    }
}

static inline uint8_t ati_3d_clamp8(float f)
{
    int i = (int)(f * 255.0f + 0.5f);

    return i < 0 ? 0 : (i > 255 ? 255 : i);
}

/*
 * Texture sampling.
 *
 * TEX_SIZE_PITCH_C holds log2 values in nibbles: pitch at bit 0, width at
 * bit 4, height at bit 8. PRIM_TEX_CNTL_C carries the texel format in bits
 * 19:16. PRIM_TEX_0_OFFSET_C is the base address in video memory.
 *
 * Point sampling only, and the wrap mode is not decoded - coordinates are
 * simply masked, which is what a repeating texture needs and what nearly
 * every surface uses.
 */
typedef struct ATI3DTex {
    bool valid;
    uint32_t tile_mode;
    bool in_gart;           /* texels come from AGP memory, not VRAM */
    ATIVGAState *s;
    uint32_t off;           /* base, VRAM byte offset or GART byte offset */
    const uint8_t *base;    /* VRAM case only */
    unsigned w, h, pitch;   /* pixels, pixels, bytes */
    unsigned bypp;          /* bytes per texel */
    unsigned datatype;
    /* One-page cache for the GART case - texel access is very coherent. */
    uint32_t cached_page;
    bool cache_valid;
    uint8_t page[4096];
} ATI3DTex;

/*
 * Fetch "n" bytes of texel data. Textures may live in video memory or, as
 * the Mac OS X driver does for Quake 3, in AGP memory reached through the
 * GART - the same place the vertex buffers come from. Reading AGP offsets
 * straight out of VRAM returned whatever else happened to be there, which
 * is why unrelated on-screen content appeared as the texture.
 */
static const uint8_t *ati_3d_texel_ptr(ATI3DTex *t, uint32_t off, unsigned n)
{
    uint32_t page;
    hwaddr pa;

    if (!t->in_gart) {
        if (off + n > t->s->vga.vram_size) {
            return NULL;
        }
        return t->s->vga.vram_ptr + off;
    }

    page = off & ~0xfffU;
    if ((off & 0xfff) + n > 4096) {
        return NULL;          /* straddles a page; skip this texel */
    }
    if (!t->cache_valid || page != t->cached_page) {
        if (!ati_gart_translate(t->s, page, &pa)) {
            return NULL;
        }
        pci_dma_read(&t->s->dev, pa, t->page, sizeof(t->page));
        t->cached_page = page;
        t->cache_valid = true;
    }
    return t->page + (off & 0xfff);
}

/*
 * Shared setup for both texture units. "raw_offset" is the unit's own
 * PRIM_TEX_0_OFFSET_C or SEC_TEX_0_OFFSET_C; size/height/pitch shifts pick
 * the unit's own fields out of TEX_SIZE_PITCH_C (primary in the low 16
 * bits, secondary in the high 16, per R128_SEC_TEX_SIZE_PITCH_SHIFT);
 * "enabled" is the unit's own enable bit (TEX_CNTL_C for primary,
 * SEC_TEX_CNTL_C's own enable for secondary); "cntl_datatype" is the
 * PRIM_TEX_CNTL_C-style word this unit takes its pixel format from - both
 * units use the same datatype field layout, but Rage128 only has one such
 * field (in PRIM_TEX_CNTL_C); the secondary unit is assumed to share the
 * primary's format, which is what every draw observed so far does.
 */
static void ati_3d_tex_setup_raw(ATIVGAState *s, ATI3DTex *t,
                                 uint32_t raw_offset, bool enabled,
                                 unsigned size_shift, unsigned height_shift,
                                 unsigned pitch_shift)
{
    uint32_t sp = s->regs.tex_size_pitch;
    /*
     * Bits 31:30 are the tiling mode (R128_TEX_NO_TILE / TILED_BY_HOST /
     * TILED_BY_STORAGE / TILED_BY_STORAGE2), not part of the address - only
     * masking the low 5 bits left them in, so a tiled texture's offset was
     * corrupted by up to 0xC0000000. Tiled layouts aren't implemented; mask
     * the mode out and read as linear, which is wrong for a genuinely tiled
     * texture but at least stops the address from being garbage.
     */
    uint32_t off = raw_offset & 0x3fffffe0UL;

    t->tile_mode = (raw_offset >> 30) & 0x3;
    if (t->tile_mode != 0) {
        qemu_log_mask(LOG_UNIMP,
            "ati_3d: tiled texture (mode %u) not implemented, sampling as "
            "linear\n", t->tile_mode);
    }

    /*
     * Clear geometry too, not just the valid flag: a caller that ignores
     * ->valid, or any future path that reads these before checking it,
     * must not see another texture's dimensions.
     */
    t->valid = false;
    t->w = t->h = t->pitch = 0;
    t->base = NULL;
    t->in_gart = false;
    if (!enabled) {
        return;
    }
    t->w     = 1u << ((sp >> size_shift) & 0xf);
    t->h     = 1u << ((sp >> height_shift) & 0xf);
    t->datatype = (s->regs.prim_tex_cntl & TEX_DATATYPE_MASK)
                  >> TEX_DATATYPE_SHIFT;

    switch (t->datatype) {
    case TEX_DATATYPE_ARGB8888:
        t->bypp = 4;
        break;
    case TEX_DATATYPE_RGB565:
    case TEX_DATATYPE_ARGB1555:
        t->bypp = 2;
        break;
    case TEX_DATATYPE_CI8:
    case TEX_DATATYPE_RGB332:
    case TEX_DATATYPE_Y8:
        t->bypp = 1;
        break;
    default:
        return;
    }

    /*
     * The pitch nibble is log2 of the pitch in PIXELS, so the byte stride
     * is that times the texel size. Treating it as bytes read every row a
     * quarter of the way along for 32bpp textures, which both striped the
     * image and walked off into whatever else was in video memory.
     */
    t->pitch = (1u << ((sp >> pitch_shift) & 0xf)) * t->bypp;

    if (!t->w || !t->h || !t->pitch) {
        return;
    }
    t->s = s;
    t->off = off;
    t->cache_valid = false;

    /*
     * Decide where the texture lives.
     *
     * Order matters, and the rule is: video memory whenever the texture
     * can be there, GART only when it cannot.
     *
     *  1. A recorded upload is ground truth - some write actually landed
     *     on this page, so use whichever memory it went to.
     *  2. Otherwise prefer video memory, provided the texture fits and the
     *     address is not inside a display surface. Most textures are
     *     written by the guest CPU straight through the linear aperture,
     *     which leaves no trace for the cache to record: gltest2 does
     *     19038 correct 8x8 samples from 0x494000 in a run with zero host
     *     data blits and no 2D blit writing that address.
     *  3. Only fall through to the GART when video memory is impossible -
     *     the texture would not fit, or the address belongs to a
     *     framebuffer. Video memory is never swapped out from under us,
     *     so a texture that fits there needs no GART lookup at all.
     *
     * The display-surface test is what keeps Quartz out of the scene. Those
     * window backing stores sit at exactly the offsets textures use, and
     * sampling one paints the desktop onto geometry.
     */
    {
        uint32_t page = off & ~0xfffU;
        bool known_in_gart;
        bool fits_vram = off + (size_t)t->h * t->pitch <= s->vga.vram_size;
        bool is_display = ati_addr_in_display_buffer(s, off);

        if (ati_tex_loc_lookup(s, page, &known_in_gart)) {
            t->in_gart = known_in_gart;
        } else if (fits_vram && !is_display) {
            t->in_gart = false;
        } else {
            hwaddr pa;

            t->in_gart = ati_gart_translate(s, page, &pa);
        }

        if (!t->in_gart) {
            if (!fits_vram || is_display) {
                return;
            }
            t->base = s->vga.vram_ptr + off;
        }
    }
    t->valid = true;
}

static void ati_3d_texel(ATI3DTex *t, float fs, float ft,
                         float *r, float *g, float *b, float *a)
{
    /*
     * Wrap properly. Texture coordinates routinely fall outside 0..1 for
     * repeating surfaces, and casting a negative float straight to unsigned
     * is undefined - in practice it yields a huge value that the mask then
     * scatters anywhere in memory. With textures packed back to back that
     * meant sampling whichever one happened to sit next door, so artwork
     * from one surface bled into another.
     */
    int ix = (int)floorf(fs * (float)t->w);
    int iy = (int)floorf(ft * (float)t->h);
    unsigned x = (unsigned)(ix & (int)(t->w - 1));
    unsigned y = (unsigned)(iy & (int)(t->h - 1));
    const uint8_t *p;

    *a = 1.0f;
    *r = *g = *b = 1.0f;

    p = ati_3d_texel_ptr(t, t->off + (uint32_t)y * t->pitch + x * t->bypp,
                         t->bypp);
    if (!p) {
        return;
    }
    p -= (size_t)x * t->bypp;      /* the cases below index by x */
    switch (t->datatype) {
    case TEX_DATATYPE_ARGB8888: {
        uint32_t c = ldl_be_p(p + x * 4);

        /*
         * Texels are R,G,B,A in memory.
         *
         * A controlled test settles this: gltest2 uploads a texture whose
         * exact bytes are known. Red is FF,00,00,FF, which read big-endian
         * is 0xFF0000FF - decoding that as A,R,G,B yields blue, and the
         * blue row 00,00,FF,FF yields cyan with zero alpha. Both of those
         * wrong colours are exactly what appeared on screen, and black
         * (00,00,00,FF) came out fully transparent, which is why the
         * checkerboard collapsed into flat bands instead of alternating.
         *
         * An earlier sample of Quake 3 data looked like opaque ARGB, but
         * that was read from an address later shown to hold a Quartz window
         * surface rather than texture data, so it proved nothing.
         */
        *r = ((c >> 24) & 0xff) / 255.0f;
        *g = ((c >> 16) & 0xff) / 255.0f;
        *b = ((c >> 8) & 0xff) / 255.0f;
        *a = (c & 0xff) / 255.0f;
        break;
    }
    case TEX_DATATYPE_RGB565: {
        uint16_t c = lduw_be_p(p + x * 2);

        *r = ((c >> 11) & 0x1f) / 31.0f;
        *g = ((c >> 5) & 0x3f) / 63.0f;
        *b = (c & 0x1f) / 31.0f;
        break;
    }
    case TEX_DATATYPE_ARGB1555: {
        uint16_t c = lduw_be_p(p + x * 2);

        *a = (c & 0x8000) ? 1.0f : 0.0f;
        *r = ((c >> 10) & 0x1f) / 31.0f;
        *g = ((c >> 5) & 0x1f) / 31.0f;
        *b = (c & 0x1f) / 31.0f;
        break;
    }
    case TEX_DATATYPE_Y8:
        *r = *g = *b = p[x] / 255.0f;
        break;
    default:
        *r = *g = *b = 1.0f;
        break;
    }
}

/* Resolve one of the ALPHA_BLEND_* factors into per-channel multipliers. */
static void ati_3d_blend_factor(unsigned fn,
                                float sr, float sg, float sb, float sa,
                                float dr, float dg, float db, float da,
                                float *fr, float *fg, float *fb)
{
    switch (fn) {
    case ALPHA_BLEND_ZERO:
        *fr = *fg = *fb = 0.0f;
        break;
    case ALPHA_BLEND_SRCCOLOR:
        *fr = sr; *fg = sg; *fb = sb;
        break;
    case ALPHA_BLEND_INVSRCCOLOR:
        *fr = 1.0f - sr; *fg = 1.0f - sg; *fb = 1.0f - sb;
        break;
    case ALPHA_BLEND_SRCALPHA:
        *fr = *fg = *fb = sa;
        break;
    case ALPHA_BLEND_INVSRCALPHA:
        *fr = *fg = *fb = 1.0f - sa;
        break;
    case ALPHA_BLEND_DSTALPHA:
        *fr = *fg = *fb = da;
        break;
    case ALPHA_BLEND_INVDSTALPHA:
        *fr = *fg = *fb = 1.0f - da;
        break;
    case ALPHA_BLEND_DSTCOLOR:
        *fr = dr; *fg = dg; *fb = db;
        break;
    case ALPHA_BLEND_INVDSTCOLOR:
        *fr = 1.0f - dr; *fg = 1.0f - dg; *fb = 1.0f - db;
        break;
    case ALPHA_BLEND_ONE:
    default:
        *fr = *fg = *fb = 1.0f;
        break;
    }
}

static void ati_3d_tex0_setup(ATIVGAState *s, ATI3DTex *t)
{
    ati_3d_tex_setup_raw(s, t, s->regs.prim_tex_offset,
                         s->regs.tex_cntl != 0,
                         TEX_SIZE_SHIFT, TEX_HEIGHT_SHIFT, TEX_PITCH_SHIFT);
}

static void ati_3d_tex1_setup(ATIVGAState *s, ATI3DTex *t)
{
    ati_3d_tex_setup_raw(s, t, s->regs.sec_tex_offset,
                         s->regs.sec_tex_cntl != 0,
                         SEC_TEX_SIZE_SHIFT, SEC_TEX_HEIGHT_SHIFT,
                         SEC_TEX_PITCH_SHIFT);
}

/*
 * Combine the two texture stages into one RGBA result. SEC_TEX_COMBINE_CNTL_C
 * has the same low-nibble op encoding as the primary combine register
 * (COMB_MODULATE etc), applied here between the primary's own output and the
 * secondary sample. Only modulate and replace are implemented; anything else
 * falls back to modulate, which is the overwhelmingly common case and safe -
 * it just won't be exactly right for a shader using add/blend stages.
 */
static void ati_3d_combine_sec(uint32_t combine_cntl,
                               float *r, float *g, float *b, float *a,
                               float sr, float sg, float sb, float sa)
{
    switch (combine_cntl & 0xf) {
    case 0x0:                          /* replace */
        *r = sr; *g = sg; *b = sb; *a = sa;
        break;
    default:                           /* modulate, and unhandled ops */
        *r *= sr; *g *= sg; *b *= sb; *a *= sa;
        break;
    }
}

/*
 * Depth buffer support.
 *
 * Z_OFFSET_C / Z_PITCH_C give the buffer, Z_STEN_CNTL_C the pixel width and
 * compare function, and TEX_CNTL_C bits 0 and 1 enable testing and writing
 * (all per r128_reg.h). Without this the rasterizer drew in submission
 * order, so overlapping objects - glxgears' three gears, a level's far and
 * near walls - resolved by whichever happened to be drawn last.
 */
typedef struct ATI3DZBuf {
    bool test;              /* Z_ENABLE */
    bool write;             /* Z_WRITE_ENABLE */
    unsigned func;          /* Z_TEST_* compare, already shifted down */
    unsigned bypp;          /* 2 for 16-bit, 4 for 24/32-bit */
    uint8_t *base;
    unsigned pitch;         /* bytes per row */
    size_t limit;           /* bytes available from base to end of VRAM */
} ATI3DZBuf;

static void ati_3d_zbuf_setup(ATIVGAState *s, ATI3DZBuf *z)
{
    unsigned width_sel;

    z->test = (s->regs.tex_cntl & TEX_CNTL_Z_ENABLE) != 0;
    z->write = (s->regs.tex_cntl & TEX_CNTL_Z_WRITE_ENABLE) != 0;
    if (!z->test && !z->write) {
        return;
    }

    z->func = (s->regs.z_sten_cntl & Z_TEST_MASK) >> Z_TEST_SHIFT;
    width_sel = (s->regs.z_sten_cntl & Z_PIX_WIDTH_MASK) >> Z_PIX_WIDTH_SHIFT;
    z->bypp = (width_sel == 0) ? 2 : 4;
    /*
     * Z_PITCH_C counts 8-PIXEL units, so the byte stride depends on the
     * depth format's own width - 8 * bypp, not a flat 32.
     *
     * Checked against two captures. A 32-bit depth run has Z_PITCH 0x26
     * (38) with a 1216-byte colour pitch: 38 * 8 * 4 = 1216, and both
     * buffers are 304 pixels wide. A 16-bit depth run has 0x64 (100) with
     * a 3200-byte colour pitch: 100 * 8 * 2 = 1600 bytes, 800 pixels -
     * matching the 800-pixel colour buffer, where a flat * 32 would have
     * made the depth buffer 1600 pixels wide and desynchronised every row.
     */
    z->pitch = (s->regs.z_pitch & Z_PITCH_MASK) * 8 * z->bypp;

    if (!z->pitch) {
        z->test = z->write = false;
        return;
    }
    if ((size_t)s->regs.z_offset + z->pitch > s->vga.vram_size) {
        z->test = z->write = false;
        return;
    }
    z->base = s->vga.vram_ptr + s->regs.z_offset;
    z->limit = s->vga.vram_size - s->regs.z_offset;

    {
        static bool logged;

        if (!logged) {
            logged = true;
            qemu_log("ati_3d: zbuffer test=%d write=%d func=%u bypp=%u "
                     "off=0x%x pitch=%u\n",
                     z->test, z->write, z->func, z->bypp,
                     s->regs.z_offset, z->pitch);
        }
    }
}

/*
 * Returns true if the fragment passes, and stores the new depth when the
 * test passes and writing is enabled.
 */
static bool ati_3d_ztest(ATI3DZBuf *z, unsigned x, unsigned y, float zf)
{
    uint8_t *p;
    uint32_t zmax, znew, zold;
    bool pass;

    if (!z->test && !z->write) {
        return true;
    }
    /* Stay inside video memory; a bad offset must not fault the host. */
    if ((size_t)y * z->pitch + (size_t)(x + 1) * z->bypp > z->limit) {
        return true;
    }
    p = z->base + (size_t)y * z->pitch + (size_t)x * z->bypp;

    zmax = (z->bypp == 2) ? 0xffff : 0x00ffffff;
    zf = (zf < 0.0f) ? 0.0f : (zf > 1.0f ? 1.0f : zf);
    znew = (uint32_t)(zf * (float)zmax);

    zold = (z->bypp == 2) ? lduw_be_p(p) : (ldl_be_p(p) & 0x00ffffff);

    /*
     * An all-zero depth buffer means it has never been cleared as far as
     * we can tell. The guest clears depth with CPU writes through the
     * linear aperture, which leave no trace here, and if that clear does
     * not reach the memory we read then every fragment fails a LESS test
     * against zero and the whole scene goes black - measured: znew=43689
     * against zold=0 for every pixel of gltest2.
     *
     * Treat a zero slot as "nothing has been drawn here yet" and let the
     * fragment through, still writing its depth. Once real depths are in
     * the buffer the comparison behaves normally, so objects still sort
     * against each other; only the very first fragment at each pixel gets
     * a free pass, which is exactly what a cleared buffer would give it.
     */
    if (zold == 0) {
        if (z->write) {
            if (z->bypp == 2) {
                stw_be_p(p, znew);
            } else {
                stl_be_p(p, (ldl_be_p(p) & 0xff000000) | znew);
            }
        }
        return true;
    }

    switch (z->func) {
    case 0:  pass = false;                break;  /* NEVER */
    case 1:  pass = znew <  zold;         break;  /* LESS */
    case 2:  pass = znew <= zold;         break;  /* LESSEQUAL */
    case 3:  pass = znew == zold;         break;  /* EQUAL */
    case 4:  pass = znew >= zold;         break;  /* GREATEREQUAL */
    case 5:  pass = znew >  zold;         break;  /* GREATER */
    case 6:  pass = znew != zold;         break;  /* NEQUAL */
    default: pass = true;                 break;  /* ALWAYS */
    }
    if (!z->test) {
        pass = true;
    }
    if (pass && z->write) {
        if (z->bypp == 2) {
            stw_be_p(p, znew);
        } else {
            stl_be_p(p, (ldl_be_p(p) & 0xff000000) | znew);
        }
    }
    return pass;
}

static void ati_3d_tri(ATIVGAState *s, const ATI3DVert *a,
                       const ATI3DVert *b, const ATI3DVert *c,
                       bool textured, bool has_s2t2)
{
    /*
     * dst_pitch is in units of 8 pixels, so at 32bpp the byte stride is
     * pitch * 8 pixels * 4 bytes = pitch * 32 - the same "* bpp" the 2D
     * engine applies. Dividing by 8 as well squeezed every row into an
     * eighth of its width and piled the image into a band at the top.
     */
    /*
     * The CCE context's target (DST_PITCH_OFFSET_C), not whatever a plain
     * MMIO DST_PITCH_OFFSET write last left behind - see ati_int.h. Fall
     * back to the shared state if _C has never been programmed, so a guest
     * that only uses the MMIO register still renders somewhere sane rather
     * than at offset 0 with a zero pitch.
     */
    unsigned pitch = s->regs.dst_pitch_3d ?
                     s->regs.dst_pitch_3d * 32 : s->regs.dst_pitch * 32;
    uint8_t *base = s->vga.vram_ptr + (s->regs.dst_pitch_3d ?
                     s->regs.dst_offset_3d : s->regs.dst_offset);
    int minx, maxx, miny, maxy, x, y;
    float area;
    /* Static: holds a 4 KiB page cache, too large for the stack and worth
     * keeping warm across the triangles of a single draw. */
    /*
     * Not static. These used to persist across calls, which meant any early
     * return inside the setup path - texturing disabled, a zero dimension,
     * an address that will not translate - left the PREVIOUS texture's
     * width, height and pitch in place. The new texture's memory was then
     * sampled with the old texture's stride, which is exactly the kind of
     * mismatch that renders as horizontal banding across a surface.
     *
     * gltest2 never showed it because it binds a single texture for the
     * whole run, so the stale geometry always happened to be correct.
     * Quake 3 switches textures constantly - 197800 draws reported 512x512
     * while the size register was being programmed with 8x8, 16x16, 32x32,
     * 64x64, 128x128, 128x64 and 256x256 for those same draws.
     */
    ATI3DTex tex, tex2;
    ATI3DZBuf zbuf = { 0 };
    bool blend, use_sec;
    unsigned src_fn, dst_fn;

    if (!pitch) {
        return;
    }

    area = (b->x - a->x) * (c->y - a->y) - (c->x - a->x) * (b->y - a->y);
    if (area == 0.0f) {
        return;                       /* degenerate */
    }

    /*
     * Backface culling is DISABLED.
     *
     * The per-frame counters showed all 12 triangles of a cube sharing one
     * winding sign and flipping together (12 kept then 12 culled), which is
     * impossible for a real solid - a cube always presents a mix. That means
     * the geometry reaching here is already flat, so culling on winding was
     * both meaningless and destructive: once the sign flipped it discarded
     * entire frames. The flattening is upstream; fix that first, then
     * revisit culling with geometry that actually has depth.
     *
     * Instead, log the real vertex positions for the first few triangles so
     * the flattening can be located rather than guessed at. If a cube's 12
     * triangles all report nearly the same z, or z that never varies with
     * rotation, the depth is being lost before the rasterizer sees it.
     */
    /*
     * Backface culling, driven by PM4_VC_FPU_SETUP: bit 0 is the front-face
     * winding (CW/CCW) and bits 1-2 the backface mode, with 0 meaning cull.
     *
     * This was disabled earlier because it discarded entire frames - but
     * that was measured against truncated geometry, where only one face of
     * any object ever arrived so every triangle shared a winding. With whole
     * command buffers now decoding, winding genuinely varies per triangle
     * and culling is both meaningful and necessary: without it, and with no
     * depth buffer, back faces paint over front ones in submission order.
     *
     * The running tally below is the check on the sign convention. For
     * closed objects roughly half of all triangles face away from the
     * camera, so culled should track near 50% of submitted. A figure near
     * 0% or 100% means the comparison is inverted and the fix is flipping
     * the area test - not further investigation.
     */
    {
        unsigned backface_mode = (s->regs.vc_fpu_setup >> 1) & 0x3;
        bool front_is_ccw = (s->regs.vc_fpu_setup & 1) != 0;
        bool is_front = front_is_ccw ? (area < 0.0f) : (area > 0.0f);
        bool cull = (backface_mode == 0 && !is_front);
        static unsigned long submitted, culled;

        submitted++;
        if (cull) {
            culled++;
        }
        /*
         * Log the first few culling decisions with their winding, so an
         * inverted sign is visible directly rather than inferred from a
         * ratio. For a closed object both signs must appear; if every
         * culled triangle shares one sign and every kept one the other,
         * that is correct - if the kept set is the one facing away, the
         * comparison below is backwards.
         */
        {
            static unsigned n_dec;

            if (n_dec < 12 && backface_mode == 0) {
                n_dec++;
                qemu_log("ati_3d: cull decision area=%+.1f front=%d "
                         "cull=%d\n", area, is_front, cull);
            }
        }
        if ((submitted % 20000) == 0) {
            qemu_log("ati_3d: culling: %lu submitted, %lu culled (%lu%%) "
                     "front_ccw=%d mode=%u\n",
                     submitted, culled, (culled * 100) / submitted,
                     front_is_ccw, backface_mode);
        }
        if (cull) {
            return;
        }
    }

    /*
     * Only sample a texture when the vertex format actually carries s,t.
     * TEX_CNTL_C is left set from an earlier draw, so keying off it alone
     * made untextured geometry multiply by the texel at (0,0) - which for
     * most textures is black, and blanked the drawing entirely.
     *
     * The secondary unit is sampled whenever SEC_TEX_CNTL_C's own enable
     * bit is set - this was never implemented at all before, meaning any
     * multi-textured surface (a base layer plus a mask, glow or lightmap,
     * which is standard for UI shaders) rendered with only the primary
     * layer and no second stage composited on top.
     */
    tex.valid = false;
    tex2.valid = false;
    if (textured) {
        ati_3d_tex0_setup(s, &tex);
        /*
         * Gate on has_s2t2 (this triangle's own vertex format carrying
         * VC_FRMT_S2_T2), not just SEC_TEX_CNTL_C being nonzero. The
         * register is left set from whatever multi-textured draw last
         * touched it and never clears, so keying off it alone enabled
         * secondary sampling on later draws that never asked for it -
         * s2,t2 defaulted to (0,0), and modulating by whatever texel sits
         * at the corner of an unrelated texture blacked out the result.
         * This is the same mistake already fixed once for the primary
         * unit's TEX_CNTL_C, made again here for the secondary one.
         */
        if (s->regs.sec_tex_cntl && has_s2t2) {
            ati_3d_tex1_setup(s, &tex2);
        }
    }
    use_sec = tex2.valid;

    ati_3d_zbuf_setup(s, &zbuf);

    /*
     * One-shot log of the setup-engine registers. SETUP_CNTL and
     * PM4_VC_FPU_SETUP are decoded values, not raw bytes to interpret -
     * bit 0 of VC_FPU_SETUP is the front-face winding (CW/CCW), bits 1-2
     * and 3-4 are the backface/frontface cull modes, and bit 9 of
     * SETUP_CNTL selects whether S,T arrive needing a divide-by-W or
     * already final. Both registers are stored now but not yet acted on -
     * this measures what the driver actually configures before any of it
     * gets implemented, rather than guessing an interpretation.
     */
    {
        static bool logged_setup;

        /*
         * SETUP_CNTL bit 19 selects subpixel precision: SUB_PIX_2BITS (0)
         * or SUB_PIX_4BITS (1), per r128_reg.h. This has never been read,
         * and Tux Racer sets it (SETUP_CNTL=0x80220) while Quake 3 and
         * gltest2 do not - which lines up with Tux Racer being the one
         * app whose vertices arrive as ~0.016 instead of screen pixels.
         * Log what each candidate scaling would produce for a real vertex
         * so the right transform can be read off rather than guessed.
         */
        if (!logged_setup) {
            unsigned subpix = (s->regs.setup_cntl >> 19) & 1;

            qemu_log("ati_3d: SUB_PIX_%uBITS  raw v0=(%.6f,%.6f)  "
                     "x16=(%.3f,%.3f)  x256=(%.3f,%.3f)  "
                     "x4096=(%.3f,%.3f)\n",
                     subpix ? 4 : 2, a->x, a->y,
                     a->x * 16.0f, a->y * 16.0f,
                     a->x * 256.0f, a->y * 256.0f,
                     a->x * 4096.0f, a->y * 4096.0f);
        }
        if (!logged_setup) {
            uint32_t sc = s->regs.setup_cntl;
            uint32_t fpu = s->regs.vc_fpu_setup;

            logged_setup = true;
            qemu_log("ati_3d: SETUP_CNTL=0x%08x VC_FPU_SETUP=0x%08x  "
                     "front_dir=%s  backface=%u frontface=%u  "
                     "st_mode=%s  color_mode=%u prim_type=%u\n",
                     sc, fpu,
                     (fpu & 1) ? "CCW" : "CW",
                     (fpu >> 1) & 0x3, (fpu >> 3) & 0x3,
                     (sc & (1 << 9)) ? "DIRECT" : "MULT_W",
                     (sc >> 3) & 0x7, (sc >> 7) & 0x3);
        }
    }

    /*
     * Per-draw address trace. Logs the raw and masked PRIM_TEX_0_OFFSET_C,
     * the resolved host address (VRAM offset or GART physical address), and
     * the first texel, for every textured draw up to the cap. Meant to be
     * captured as two SEPARATE short sessions - one that quits right after
     * the menu is visible, one that goes straight into a timedemo - so the
     * two address ranges in use can be compared directly instead of guessed
     * apart from a single combined log.
     */
    {
        static unsigned n_tex_log;
        static unsigned seq;

        seq++;
        if (textured && tex.valid && n_tex_log < 200000) {
            hwaddr pa = 0;
            bool resolved;

            n_tex_log++;

            if (tex.in_gart) {
                resolved = ati_gart_translate(s, tex.off & ~0xfffU, &pa);
                pa += tex.off & 0xfff;
            } else {
                resolved = true;
                pa = tex.off;
            }
            qemu_log("ati_3d: #%u raw_off=0x%x masked_off=0x%x tile=%u "
                     "%s pa=0x%llx resolved=%d w=%u h=%u\n",
                     seq, s->regs.prim_tex_offset, tex.off, tex.tile_mode,
                     tex.in_gart ? "gart" : "vram",
                     (unsigned long long)pa, resolved, tex.w, tex.h);
        }
    }

    blend = (s->regs.tex_cntl & TEX_CNTL_ALPHA_ENABLE) != 0;
    src_fn = (s->regs.misc_3d_state >> ALPHA_BLEND_SRC_SHIFT)
             & ALPHA_BLEND_MASK;
    dst_fn = (s->regs.misc_3d_state >> ALPHA_BLEND_DST_SHIFT)
             & ALPHA_BLEND_MASK;

    minx = (int)floorf(MIN(a->x, MIN(b->x, c->x)));
    maxx = (int)ceilf(MAX(a->x, MAX(b->x, c->x)));
    miny = (int)floorf(MIN(a->y, MIN(b->y, c->y)));
    maxy = (int)ceilf(MAX(a->y, MAX(b->y, c->y)));

    /* Clip to the scissor rectangle. */
    minx = MAX(minx, s->regs.sc_left);
    miny = MAX(miny, s->regs.sc_top);
    maxx = MIN(maxx, s->regs.sc_right);
    maxy = MIN(maxy, s->regs.sc_bottom);
    if (minx < 0) {
        minx = 0;
    }
    if (miny < 0) {
        miny = 0;
    }

    for (y = miny; y <= maxy; y++) {
        uint8_t *row = base + (size_t)y * pitch;

        if (row + (size_t)(maxx + 1) * 4 >
            s->vga.vram_ptr + s->vga.vram_size) {
            break;
        }
        for (x = minx; x <= maxx; x++) {
            float px = x + 0.5f, py = y + 0.5f;
            float w0, w1, w2;
            uint32_t pix;

            w0 = ((b->x - a->x) * (py - a->y) -
                  (px - a->x) * (b->y - a->y)) / area;
            w1 = ((c->x - b->x) * (py - b->y) -
                  (px - b->x) * (c->y - b->y)) / area;
            w2 = 1.0f - w0 - w1;

            float sr, sg, sb, sa;

            if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) {
                continue;
            }

            /*
             * Depth test before shading: a fragment that loses here costs
             * nothing further, and texture sampling is the expensive part.
             * Interpolated with the same barycentrics as everything else.
             */
            if (zbuf.test || zbuf.write) {
                float pz = w1 * a->z + w2 * b->z + w0 * c->z;

                if (!ati_3d_ztest(&zbuf, x, y, pz)) {
                    continue;
                }
            }

            /* w1 weights a, w2 weights b, w0 weights c */
            sr = w1 * a->r + w2 * b->r + w0 * c->r;
            sg = w1 * a->g + w2 * b->g + w0 * c->g;
            sb = w1 * a->b + w2 * b->b + w0 * c->b;
            sa = w1 * a->a + w2 * b->a + w0 * c->a;

            /*
             * Modulate by the texture. The combine mode in
             * PRIM_TEXTURE_COMBINE_CNTL_C is not decoded yet; modulate is
             * what the overwhelming majority of surfaces ask for, and it
             * reduces to the texel alone when the vertex colour is white.
             */
            if (tex.valid) {
                float ts = w1 * a->s + w2 * b->s + w0 * c->s;
                float tt = w1 * a->t + w2 * b->t + w0 * c->t;
                float tr, tg, tb, ta;

                ati_3d_texel(&tex, ts, tt, &tr, &tg, &tb, &ta);
                sr *= tr;
                sg *= tg;
                sb *= tb;
                sa *= ta;
            }

            /*
             * Second texture stage. SEC_TEX_CNTL_C selects whether it
             * samples the primary unit's own s,t (SEC_SELECT_PRIM_ST) or a
             * separate s2,t2 pair carried in the vertex (SEC_SELECT_SEC_ST,
             * VC_FRMT_S2_T2) - the vertex format decode already extracts
             * both, this is the first place either gets used.
             */
            if (use_sec) {
                bool sec_st = (s->regs.sec_tex_cntl & SEC_SELECT_SEC_ST) != 0;
                float ts2 = sec_st
                    ? (w1 * a->s2 + w2 * b->s2 + w0 * c->s2)
                    : (w1 * a->s + w2 * b->s + w0 * c->s);
                float tt2 = sec_st
                    ? (w1 * a->t2 + w2 * b->t2 + w0 * c->t2)
                    : (w1 * a->t + w2 * b->t + w0 * c->t);
                float tr2, tg2, tb2, ta2;

                ati_3d_texel(&tex2, ts2, tt2, &tr2, &tg2, &tb2, &ta2);
                ati_3d_combine_sec(s->regs.sec_tex_combine,
                                   &sr, &sg, &sb, &sa, tr2, tg2, tb2, ta2);
            }

            /*
             * Alpha blending, with the factors taken from
             * MISC_3D_STATE_CNTL_REG rather than assumed. Quake 3 uses
             * several: SRCALPHA/INVSRCALPHA for translucent overlays,
             * ONE/ONE for additive stages, DSTCOLOR/ONE for modulated
             * ones. Treating every stage as source-over composited the
             * additive passes as if they were translucent, which lost
             * most of the brightness in the menus.
             */
            if (blend) {
                uint32_t dst = s->vga.big_endian_fb ?
                               ldl_be_p(row + (size_t)x * 4) :
                               ldl_le_p(row + (size_t)x * 4);
                float dr = ((dst >> 16) & 0xff) / 255.0f;
                float dg = ((dst >> 8) & 0xff) / 255.0f;
                float db = (dst & 0xff) / 255.0f;
                float da = ((dst >> 24) & 0xff) / 255.0f;
                float sf_r, sf_g, sf_b, df_r, df_g, df_b;

                ati_3d_blend_factor(src_fn, sr, sg, sb, sa, dr, dg, db, da,
                                    &sf_r, &sf_g, &sf_b);
                ati_3d_blend_factor(dst_fn, sr, sg, sb, sa, dr, dg, db, da,
                                    &df_r, &df_g, &df_b);

                sr = sr * sf_r + dr * df_r;
                sg = sg * sf_g + dg * df_g;
                sb = sb * sf_b + db * df_b;
                sr = MIN(sr, 1.0f);
                sg = MIN(sg, 1.0f);
                sb = MIN(sb, 1.0f);
            }

            pix = 0xff000000u |
                  ((uint32_t)ati_3d_clamp8(sr) << 16) |
                  ((uint32_t)ati_3d_clamp8(sg) << 8) |
                   (uint32_t)ati_3d_clamp8(sb);
            /*
             * Store in the framebuffer's byte order, not the host's.
             * With big_endian_fb set - which it is for Mac OS X on PPC -
             * a host-endian store on a little-endian host reverses the
             * channels, turning warm colours cold by swapping red and
             * blue. The 2D engine never hit this because it copies bytes
             * without interpreting them as pixels.
             */
            if (s->vga.big_endian_fb) {
                stl_be_p(row + (size_t)x * 4, pix);
            } else {
                stl_le_p(row + (size_t)x * 4, pix);
            }
        }
    }

    if (maxy >= miny) {
        ram_addr_t off = ((s->regs.dst_pitch_3d ? s->regs.dst_offset_3d :
                           s->regs.dst_offset) + (size_t)miny * pitch);
        ram_addr_t len = (size_t)(maxy - miny + 1) * pitch;

        if (off < s->vga.vram_size) {
            memory_region_set_dirty(&s->vga.vram, off,
                                    MIN(len, s->vga.vram_size - off));
        }
    }
}

/*
 * Pull "n" vertices out of a GART-resident vertex buffer into a local array.
 * Used by the indexed primitive form, where the vertices do not travel in
 * the packet at all - only a GART offset does.
 */
static bool ati_3d_fetch_buffer(ATIVGAState *s, uint32_t gart_off,
                                unsigned first, unsigned n,
                                const ATI3DLayout *l, ATI3DVert *out)
{
    unsigned i;

    for (i = 0; i < n; i++) {
        uint32_t raw[32];
        uint32_t off = gart_off + (first + i) * l->stride * 4;
        unsigned got = 0;

        if (l->stride > ARRAY_SIZE(raw)) {
            return false;
        }
        /* A vertex can straddle a GART page, so fetch it piecewise. */
        while (got < l->stride) {
            uint32_t chunk = MIN(l->stride - got,
                                 (4096 - ((off + got * 4) & 0xfff)) / 4);
            hwaddr pa;

            if (!chunk || !ati_gart_translate(s, off + got * 4, &pa)) {
                return false;
            }
            pci_dma_read(&s->dev, pa, raw + got, chunk * 4);
            got += chunk;
        }
        for (unsigned k = 0; k < l->stride; k++) {
            raw[k] = le32_to_cpu(raw[k]);
        }
        ati_3d_fetch_vert(raw, l, &out[i]);
    }
    return true;
}

static void ati_3d_emit(ATIVGAState *s, unsigned prim,
                        const ATI3DVert *v, unsigned n, bool textured,
                        bool has_s2t2)
{
    unsigned i;

    switch (prim) {
    case CCE_VC_CNTL_PRIM_TYPE_TRI_LIST:
        for (i = 0; i + 2 < n; i += 3) {
            ati_3d_tri(s, &v[i], &v[i + 1], &v[i + 2], textured, has_s2t2);
        }
        break;
    case CCE_VC_CNTL_PRIM_TYPE_TRI_FAN:
        for (i = 1; i + 1 < n; i++) {
            ati_3d_tri(s, &v[0], &v[i], &v[i + 1], textured, has_s2t2);
        }
        break;
    case CCE_VC_CNTL_PRIM_TYPE_TRI_STRIP:
        /*
         * A strip alternates winding: triangle 0 is (v0,v1,v2), triangle 1
         * is (v1,v3,v2) - note the swap - triangle 2 is (v2,v3,v4), and so
         * on. Emitting every triangle in plain (i, i+1, i+2) order leaves
         * every odd one wound backwards, which inverts its facing. That is
         * invisible while nothing depends on winding, but it corrupts any
         * front/back distinction and breaks as soon as culling is enabled.
         */
        for (i = 0; i + 2 < n; i++) {
            if (i & 1) {
                ati_3d_tri(s, &v[i + 1], &v[i], &v[i + 2],
                           textured, has_s2t2);
            } else {
                ati_3d_tri(s, &v[i], &v[i + 1], &v[i + 2],
                           textured, has_s2t2);
            }
        }
        break;
    default:
        qemu_log_mask(LOG_UNIMP,
            "ati_3d: primitive type %u not implemented\n", prim);
        break;
    }
}

/*
 * Maximum vertices buffered for one draw. Was 1024, which 223 draws in a
 * single glxgears capture hit exactly - meaning they were being clipped.
 * A 16384-dword packet at the smallest sane stride can carry well over
 * 2000 vertices, so size accordingly.
 */
#define ATI_3D_MAX_VERTS 8192

/*
 * 3D_RNDR_GEN_PRIM (0x25) carries everything inline:
 *     format, vc_cntl, vertex data...
 *
 * 3D_RNDR_GEN_INDX_PRIM (0x23) keeps the vertices in a GART buffer and
 * sends only a pointer, matching what the r128 DRM emits:
 *     offset, size, format, vc_cntl, indices...
 * with the indices packed two 16-bit values per dword.
 *
 * Reading the inline layout for both is what made every indexed draw report
 * walk=0: the size field was being taken for vc_cntl.
 */
static void ati_3d_draw(ATIVGAState *s, unsigned opcode,
                        const uint32_t *data, unsigned count)
{
    bool indexed = (opcode & ~CCE_PACKET3_OP_CNTL_BIT) ==
                   CCE_PACKET3_OP_3D_RNDR_GEN_INDX_PRIM;
    unsigned hdr = indexed ? 4 : 2;
    uint32_t format, vc_cntl, gart_off = 0, bufsize = 0;
    unsigned prim, walk, nvtx;
    ATI3DLayout l;
    /*
     * Static, not stack: ATI3DVert is 40 bytes, so ATI_3D_MAX_VERTS of them
     * is ~320 KiB - far past any sane stack budget. This runs thousands of
     * times per frame, so a persistent buffer also avoids a large malloc
     * per draw. Single-threaded through the CCE decode, so sharing it is
     * safe.
     */
    static ATI3DVert v[ATI_3D_MAX_VERTS];

    if (count < hdr) {
        return;
    }
    if (indexed) {
        gart_off = data[0];
        bufsize  = data[1];
        format   = data[2];
        vc_cntl  = data[3];
    } else {
        format   = data[0];
        vc_cntl  = data[1];
    }

    prim = vc_cntl & CCE_VC_CNTL_PRIM_TYPE_MASK;
    walk = vc_cntl & CCE_VC_CNTL_PRIM_WALK_MASK;
    nvtx = vc_cntl >> CCE_VC_CNTL_NUM_SHIFT;

    ati_3d_layout(format, &l);
    /*
     * Do the perspective divide on S,T.
     *
     * This was gated on SETUP_CNTL bit 9, read as "coordinates are already
     * final" - a bit position I assumed rather than verified, and it was
     * backwards. Measured: gltest2 emits texcoords of exactly 1.0, and what
     * arrives is 0.2000 while RHW is 0.2000. Across frames the pair tracks
     * (0.1986/0.2004, 0.1972/0.2008, 0.1958/0.2011) with the product
     * holding at 1.0, so the driver is sending S*W and T*W and the divide
     * is always required. Skipping it shrank every texture coordinate by a
     * factor of W, so only the top-left corner of a texture was ever
     * sampled - which is why an 8x8 checkerboard rendered as two or three
     * flat blocks.
     */
    l.st_direct = false;
    if (!l.stride || nvtx < 3) {
        return;
    }
    /*
     * Clamp rather than drop. Rejecting the whole draw would silently lose
     * all of a large batch; clamping renders as much as fits and reports
     * the shortfall, which is both more visible and more useful than
     * nothing appearing at all.
     */
    if (nvtx > ATI_3D_MAX_VERTS) {
        qemu_log_mask(LOG_UNIMP,
            "ati_3d: draw of %u vertices exceeds the %u-vertex buffer, "
            "rendering the first %u\n",
            nvtx, (unsigned)ATI_3D_MAX_VERTS, (unsigned)ATI_3D_MAX_VERTS);
        nvtx = ATI_3D_MAX_VERTS;
    }

    if (walk == CCE_VC_CNTL_PRIM_WALK_RING) {
        /* Vertices inline in the packet. */
        unsigned avail = (count - hdr) / l.stride;
        unsigned i;

        nvtx = MIN(nvtx, avail);
        if (nvtx < 3) {
            return;
        }
        for (i = 0; i < nvtx; i++) {
            ati_3d_fetch_vert(data + hdr + i * l.stride, &l, &v[i]);
        }
        ati_3d_emit(s, prim, v, nvtx, l.off_st >= 0, l.off_s2t2 >= 0);
    } else if (walk == CCE_VC_CNTL_PRIM_WALK_LIST) {
        /* Sequential vertices from the GART buffer. */
        if (!ati_3d_fetch_buffer(s, gart_off, 0, nvtx, &l, v)) {
            return;
        }
        ati_3d_emit(s, prim, v, nvtx, l.off_st >= 0, l.off_s2t2 >= 0);
    } else if (walk == CCE_VC_CNTL_PRIM_WALK_IND) {
        /* Indices follow the header, two 16-bit values per dword. */
        unsigned i;

        for (i = 0; i < nvtx; i++) {
            uint32_t d = data[hdr + i / 2];
            unsigned idx = (i & 1) ? (d & 0xffff) : (d >> 16);

            if (hdr + i / 2 >= count) {
                return;
            }
            if (bufsize && idx >= bufsize) {
                return;
            }
            if (!ati_3d_fetch_buffer(s, gart_off, idx, 1, &l, &v[i])) {
                return;
            }
        }
        ati_3d_emit(s, prim, v, nvtx, l.off_st >= 0, l.off_s2t2 >= 0);
    } else {
        qemu_log_mask(LOG_UNIMP,
            "ati_3d: walk mode 0x%x not implemented, opcode 0x%02x "
            "fmt 0x%x\n", walk, opcode, format);
        return;
    }

    {
        static unsigned n_log;

        if (n_log < 4000) {
            n_log++;
            qemu_log("ati_3d: %s prim=%u walk=0x%x nvtx=%u stride=%u "
                     "fmt=0x%x off_st=%d st_direct=%d gart=0x%x dst=0x%x "
                     "pitch=%u v0=(%f,%f,%f) rgba=(%.2f,%.2f,%.2f,%.2f) "
                     "st0=(%.4f,%.4f) st1=(%.4f,%.4f) st2=(%.4f,%.4f) "
                     "rhw0=%.4f\n",
                     indexed ? "indx" : "inline", prim, walk, nvtx,
                     l.stride, format, l.off_st, l.st_direct,
                     gart_off, s->regs.dst_offset_3d,
                     s->regs.dst_pitch_3d * 32,
                     v[0].x, v[0].y, v[0].z,
                     v[0].r, v[0].g, v[0].b, v[0].a,
                     v[0].s, v[0].t,
                     nvtx > 1 ? v[1].s : 0.0f, nvtx > 1 ? v[1].t : 0.0f,
                     nvtx > 2 ? v[2].s : 0.0f, nvtx > 2 ? v[2].t : 0.0f,
                     v[0].rhw);
        }
    }
}

static int ati_bpp_from_datatype_bits(uint32_t dp_datatype)
{
    switch (dp_datatype & 0xf) {
    case 2:
        return 8;
    case 3:
    case 4:
        return 16;
    case 5:
        return 24;
    case 6:
        return 32;
    default:
        return 0;
    }
}

static void ati_cce_dispatch_packet3(ATIVGAState *s, uint32_t header,
                                      const uint32_t *data, unsigned count)
{
    unsigned opcode = (header >> 8) & 0xff;
    unsigned base_op = opcode & ~CCE_PACKET3_OP_CNTL_BIT;

    switch (opcode) {
    case CCE_PACKET3_OP_NOP:
        break;

    case CCE_PACKET3_OP_3D_RNDR_GEN_INDX_PRIM:
    case CCE_PACKET3_OP_3D_RNDR_GEN_PRIM:
        ati_3d_draw(s, opcode, data, count);
        break;

    case CCE_PACKET3_OP_3D_SAVE_CONTEXT:
    case CCE_PACKET3_OP_3D_PLAY_CONTEXT:
        qemu_log_mask(LOG_UNIMP,
            "ati_cce: 3D context save/restore (opcode 0x%02x) ignored - "
            "only one global 3D state is modelled\n", opcode);
        break;

    case CCE_PACKET3_OP_LOAD_PALETTE:
    case CCE_PACKET3_OP_PURGE:
    case CCE_PACKET3_OP_NEXT_VERTEX_BUNDLE:
        qemu_log_mask(LOG_UNIMP,
            "ati_cce: packet3 opcode 0x%02x not implemented\n", opcode);
        break;

    case CCE_PACKET3_OP_HOSTDATA_BLT | CCE_PACKET3_OP_CNTL_BIT:
    case CCE_PACKET3_OP_HOSTDATA_BLT:
    {
        /*
         * Host data blit: the pixels travel inline in the packet and are
         * written straight into the destination rectangle. This is how
         * textures get uploaded - dropping these left video memory blank
         * where a texture should be, so every textured surface sampled
         * zeroes and came out flat.
         *
         * Layout, from r128_state.c:
         *   DP_GUI_MASTER_CNTL
         *   (pitch << 21) | (offset >> 5)   destination pitch and offset
         *   SC_TOP_LEFT, SC_BOTTOM_RIGHT
         *   (y << 16) | x
         *   (height << 16) | width
         *   pixel data ...
         */
        uint32_t dst_po, xy, wh, pitch, off;
        unsigned x, y, w, h, i = 0, row, col, n;
        unsigned bypp;

        if (opcode & CCE_PACKET3_OP_CNTL_BIT) {
            if (!count) {
                break;
            }
            ati_mm_write(s, DP_GUI_MASTER_CNTL_C, data[i++], 4);
        }
        if (count < i + 5) {
            break;
        }
        dst_po = data[i++];
        i += 2;                       /* the two scissor dwords */
        xy = data[i++];
        wh = data[i++];

        /*
         * The pitch field is in units of 8 pixels, exactly as ati.c decodes
         * DST_PITCH_OFFSET, so the byte stride is field * 8 * bytes-per-
         * pixel. Treating it as 64-byte units made every row twice as far
         * apart as it should be, scattering the uploaded texels.
         */
        pitch = ((dst_po >> 21) & 0x3ff) * 8;    /* pixels; * bypp below */
        off   = (dst_po & 0x1fffff) << 5;        /* 32-byte units */
        x = xy & 0xffff;
        y = (xy >> 16) & 0xffff;
        w = wh & 0xffff;
        h = (wh >> 16) & 0xffff;

        bypp = ati_bpp_from_datatype_bits(s->regs.dp_datatype) / 8;
        if (!bypp || !pitch || !w || !h) {
            break;
        }
        pitch *= bypp;                           /* now a byte stride */
        if (count > i && data[i] == (w * h * bypp) / 4) {
            i++;                      /* optional dword count */
        }

        /*
         * The payload is a stream of pixels filling the destination
         * rectangle left to right, top to bottom - and a single upload is
         * split across many packets, each carrying only a slice of it. So
         * consume exactly the dwords present and stop, rather than assuming
         * the whole rectangle arrives at once: indexing the source by
         * (row * width + col) ran off the end of the payload after the
         * first few texels and left the rest of the texture blank.
         */
        n = count - i;
        {
            unsigned dw = 0;                  /* payload dwords consumed */
            unsigned per_px = bypp / 4;       /* dwords per pixel, 32bpp */

            if (!per_px) {
                per_px = 1;                   /* packed formats: 1 dword */
            }
            for (row = 0; row < h && dw < n; row++) {
                uint32_t dst = off + (uint32_t)(y + row) * pitch + x * bypp;
                uint32_t last_page = 0xffffffffu;

                for (col = 0; col < w && dw < n; col++) {
                    uint32_t a = dst + (uint32_t)col * bypp;
                    uint32_t page = a & ~0xfffU;

                    /*
                     * Host data blits write to VIDEO MEMORY, never through
                     * the GART.
                     *
                     * This used to translate the destination offset and,
                     * when it resolved, pci_dma_write the pixels into
                     * system RAM. But the GART maps the driver's command
                     * buffers, and in a Tux Racer capture those occupy
                     * 0x20..0xf9f840 while texture destinations run
                     * 0x6d0f80..0xffa720 - a large overlap. Every upload
                     * landing in that window DMA'd kilobytes of texel data
                     * straight over the driver's own command memory, which
                     * it later read back as pointers: a fixed faulting PC
                     * inside ATIRage128 with a different garbage address
                     * every run.
                     *
                     * The sampler reading textures through the GART is a
                     * separate matter and stays as it is; that the read
                     * path resolves there was never a reason for the write
                     * path to do the same.
                     */
                    if (a + 4 <= s->vga.vram_size) {
                        stl_be_p(s->vga.vram_ptr + a, data[i + dw]);
                        /*
                         * Only record this page as a texture location if
                         * it is not inside a display buffer.
                         *
                         * Some upload destinations land inside the front
                         * buffer - measured: 0x6b6480, 0x6bcf00, 0x6c79e0,
                         * 0x6d0f80 and 0x6d1f80 all sit within the
                         * 0x614d00..0x78bd00 front buffer. Recording those
                         * told the sampler "a texture lives here", so it
                         * read the page back and got whatever frame had
                         * most recently been presented - the desktop, the
                         * Activity Monitor's bar graphs - and painted it
                         * onto geometry.
                         *
                         * Leaving such a page unrecorded sends the sampler
                         * to the GART instead, which is where textures
                         * actually come from.
                         */
                        if (page != last_page &&
                            !ati_addr_in_display_buffer(s, a)) {
                            ati_tex_loc_record(s, page, false);
                            last_page = page;
                        }
                    } else {
                        dw = n;
                        break;
                    }
                    dw += per_px;
                }
            }
        }
        memory_region_set_dirty(&s->vga.vram, off,
                                MIN((size_t)h * pitch,
                                    s->vga.vram_size - off));
        {
            static unsigned n_hd;

            /* Only the wide uploads: those are the base mip levels. */
            if (w >= 64 && n_hd < 25) {
                n_hd++;
                qemu_log("ati_hostblt: dst=0x%x pitch=%u at (%u,%u) %ux%u "
                         "bypp=%u payload=%u  [tex_reg=0x%x]\n",
                         off, pitch, x, y, w, h, bypp, n,
                         s->regs.prim_tex_offset);
            }
        }
        break;
    }

    case CCE_PACKET3_OP_PAINT_MULTI | CCE_PACKET3_OP_CNTL_BIT:
    case CCE_PACKET3_OP_BITBLT_MULTI | CCE_PACKET3_OP_CNTL_BIT:
    case CCE_PACKET3_OP_PAINT_MULTI:
    case CCE_PACKET3_OP_BITBLT_MULTI:
    {
        /*
         * The multi-rect 2D ops carry a whole operation inline instead of
         * relying on register state. Mac OS X issues window drags this way,
         * so while these were dropped the dragged content was never painted.
         *
         * Payload layout, confirmed against the packets the guest emits:
         *
         *   [DP_GUI_MASTER_CNTL]  present for the CNTL_ opcodes only
         *   [SRC_PITCH_OFFSET]    if GMC_SRC_PITCH_OFFSET_CNTL is set
         *   [DST_PITCH_OFFSET]    if GMC_DST_PITCH_OFFSET_CNTL is set
         *   [DP_BRUSH_FRGD_CLR]   if the brush type is solid colour
         *    SRC_X_Y              BITBLT only
         *    DST_X_Y
         *    DST_WIDTH_HEIGHT
         *
         * The optional dwords are what makes the packet length vary - the
         * r128 DRM always sets both pitch/offset bits and so always emits
         * six dwords, while the Mac driver supplies only the source and
         * lets the destination come from the default registers, giving
         * five. Deriving the layout from the control word covers both.
         *
         * Dispatch goes through ati_mm_write() so the usual side effects
         * apply, including the write to DST_WIDTH_HEIGHT triggering the
         * blit itself.
         */
        bool is_blt = (base_op == CCE_PACKET3_OP_BITBLT_MULTI);
        unsigned trailing = is_blt ? 3 : 2;
        unsigned i = 0;
        uint32_t gmc;

        if (opcode & CCE_PACKET3_OP_CNTL_BIT) {
            if (!count) {
                break;
            }
            gmc = data[i++];
            ati_mm_write(s, DP_GUI_MASTER_CNTL_C, gmc, 4);
        } else {
            gmc = s->regs.dp_gui_master_cntl;
        }

        if (gmc & GMC_SRC_PITCH_OFFSET_CNTL) {
            if (i >= count) {
                break;
            }
            ati_mm_write(s, SRC_PITCH_OFFSET, data[i++], 4);
        }
        if (gmc & GMC_DST_PITCH_OFFSET_CNTL) {
            if (i >= count) {
                break;
            }
            ati_mm_write(s, DST_PITCH_OFFSET_C, data[i++], 4);
        }
        if ((gmc & 0xf0) == GMC_BRUSH_SOLIDCOLOR) {
            if (i >= count) {
                break;
            }
            ati_mm_write(s, DP_BRUSH_FRGD_CLR, data[i++], 4);
        }

        if (count - i < trailing) {
            qemu_log_mask(LOG_GUEST_ERROR,
                "ati_cce: 2D packet3 0x%02x: %u dwords left, need %u\n",
                opcode, count - i, trailing);
            break;
        }
        if (is_blt) {
            ati_mm_write(s, SRC_X_Y, data[i++], 4);
        }
        ati_mm_write(s, DST_X_Y, data[i++], 4);
        ati_mm_write(s, DST_WIDTH_HEIGHT, data[i++], 4);
        break;
    }

    default:
        if (base_op >= CCE_PACKET3_OP_PAINT &&
            base_op <= CCE_PACKET3_OP_SET_MODE24BPP) {
            qemu_log_mask(LOG_UNIMP,
                "ati_cce: 2D packet3 opcode 0x%02x (cntl=%d) not yet "
                "translated to the 2D engine - dropped, %u payload dwords "
                "consumed\n", opcode,
                (opcode & CCE_PACKET3_OP_CNTL_BIT) != 0, count);
        } else {
            qemu_log_mask(LOG_UNIMP,
                "ati_cce: unknown packet3 opcode 0x%02x\n", opcode);
        }
        break;
    }
}

void ati_cce_process(ATIVGAState *s)
{
    uint32_t mode = s->cce.buffer_cntl & PM4_BUFFER_CNTL_MODE_MASK;
    unsigned iterations = 0;

    if (mode == PM4_BUFFER_CNTL_NONPM4_MODE) {
        /* CCE not enabled; PM4_BUFFER_DL_WPTR writes are meaningless. */
        return;
    }
    /*
     * Process the ring in every PM4 mode, not just 192PIO.
     *
     * The mode field describes how the vertex and indirect-buffer paths are
     * fed (PIO versus bus-master), not the ring's packet format - the ring
     * always carries the same packet0/1/2/3 stream and the decode below is
     * mode-independent. Bailing out for anything but 192PIO discarded every
     * command the driver submitted through the ring while in mode 7
     * (PM4_64PIO_64VCBM_64INDBM), which is the mode Mac OS X actually
     * selects. Only a fraction of each frame's geometry survived - the
     * draws that happened to arrive via the FIFO or an indirect buffer -
     * so most of the scene simply never reached the rasterizer.
     */

    while (s->cce.rptr != s->cce.wptr) {
        uint32_t header;
        uint32_t type;

        if (++iterations > CCE_MAX_DWORDS_PER_CALL) {
            qemu_log_mask(LOG_GUEST_ERROR,
                "ati_cce: exceeded max iterations, aborting to avoid "
                "spinning (rptr=%u wptr=%u)\n", s->cce.rptr, s->cce.wptr);
            break;
        }

        header = ati_cce_ring_read(s, s->cce.rptr++);
        type = header & CCE_PACKET_TYPE_MASK;

        switch (type) {
        case CCE_PACKET0:
        {
            unsigned count = ((header & CCE_PACKET_COUNT_MASK) >> 16) + 1;
            unsigned reg = (header & CCE_PACKET0_REG_MASK) << 2;
            bool one_reg = (header & CCE_PACKET0_ONE_REG_WR) != 0;
            unsigned i;

            for (i = 0; i < count; i++) {
                uint32_t data = ati_cce_ring_read(s, s->cce.rptr++);
                unsigned dest = one_reg ? reg : reg + i * 4;
                ati_mm_write(s, dest, data, 4);
            }
            break;
        }

        case CCE_PACKET1:
        {
            unsigned reg0 = (header & CCE_PACKET1_REG0_MASK) << 2;
            unsigned reg1 = ((header & CCE_PACKET1_REG1_MASK) >> 11) << 2;
            uint32_t data0 = ati_cce_ring_read(s, s->cce.rptr++);
            uint32_t data1 = ati_cce_ring_read(s, s->cce.rptr++);

            ati_mm_write(s, reg0, data0, 4);
            ati_mm_write(s, reg1, data1, 4);
            break;
        }

        case CCE_PACKET2:
            /* No payload; pure pad/nop. */
            break;

        case CCE_PACKET3:
        {
            unsigned count = ((header & CCE_PACKET_COUNT_MASK) >> 16) + 1;
            uint32_t payload[CCE_MAX_PACKET3_PAYLOAD];
            unsigned n = MIN(count, CCE_MAX_PACKET3_PAYLOAD);
            unsigned i;

            for (i = 0; i < count; i++) {
                uint32_t data = ati_cce_ring_read(s, s->cce.rptr++);
                if (i < n) {
                    payload[i] = data;
                }
            }
            if (count > CCE_MAX_PACKET3_PAYLOAD) {
                qemu_log_mask(LOG_GUEST_ERROR,
                    "ati_cce: packet3 payload %u dwords truncated to %u "
                    "for decode (ring stays in sync)\n",
                    count, CCE_MAX_PACKET3_PAYLOAD);
            }
            ati_cce_dispatch_packet3(s, header, payload, n);
            break;
        }

        default:
            g_assert_not_reached();
        }
    }
}

/*
 * Feed one dword into a packet0/1/2/3 command stream decoder.
 *
 * The Rage128 carries the same packet format over three transports: the
 * ring in VRAM, the PIO FIFO (PM4_FIFO_DATA_EVEN/ODD), and indirect
 * buffers in system memory.  Each transport gets its own ATICCEStream so
 * a nested fetch cannot corrupt the stream it was triggered from - the
 * write to PM4_IW_INDSIZE that starts an indirect fetch is itself a
 * payload dword of an in-progress FIFO packet.
 *
 * Dispatch goes through ati_mm_write() exactly as ati_cce_process() does,
 * so register side effects (latching offsets, triggering a blt via
 * DP_GUI_MASTER_CNTL, ...) behave identically on every path.
 */
static void ati_cce_stream_dword(ATIVGAState *s, ATICCEStream *st,
                                 uint32_t data)
{
    uint32_t type;
    /*
     * Which stream is this? The FIFO uses s->cce.fifo; indirect buffers
     * each get their own heap-allocated state. Only the indirect stream
     * carries geometry, and only it is showing phantom opcodes, so the
     * header log below is restricted to it - logging the FIFO's init
     * traffic burned the entire cap before reaching anything relevant.
     */
    bool is_indirect = (st != &s->cce.fifo);

    st->stream_pos++;

    if (st->remaining == 0) {
        /* Expecting a packet header. */
        unsigned count = ((data & CCE_PACKET_COUNT_MASK) >> 16) + 1;

        st->hdr = data;
        st->index = 0;

        switch (data & CCE_PACKET_TYPE_MASK) {
        case CCE_PACKET0:
            st->reg = (data & CCE_PACKET0_REG_MASK) << 2;
            st->one_reg = (data & CCE_PACKET0_ONE_REG_WR) != 0;
            st->remaining = count;
            break;

        case CCE_PACKET1:
            st->remaining = 2;
            break;

        case CCE_PACKET2:
            /* Pad/nop, no payload - nothing to wait for. */
            break;

        case CCE_PACKET3:
            st->remaining = count;
            break;

        default:
            g_assert_not_reached();
        }

        /*
         * Log every header this decoder accepts, with the stream position
         * it was accepted at. Phantom opcodes (0xff, 0x8f, 0x03) mean the
         * decoder has lost alignment and is reading payload as headers -
         * the dword position of the last GOOD packet before the first bad
         * one shows exactly which packet's length is being mis-decoded,
         * which is the thing to fix. Capped so a long run stays readable.
         */
        {
            static unsigned n_hdr;

            if (is_indirect && n_hdr < 3000) {
                unsigned t = (data & CCE_PACKET_TYPE_MASK) >> 30;
                unsigned op = (data >> 8) & 0xff;

                n_hdr++;
                qemu_log("ati_cce: ind hdr#%u dword=%u raw=%08x type=%u "
                         "count=%u op=0x%02x\n",
                         n_hdr, st->stream_pos, data, t, count, op);
            }
        }
        return;
    }

    /* Payload dword. */
    type = st->hdr & CCE_PACKET_TYPE_MASK;
    switch (type) {
    case CCE_PACKET0:
        ati_mm_write(s, st->one_reg ? st->reg : st->reg + st->index * 4,
                     data, 4);
        break;

    case CCE_PACKET1:
        if (st->index < 2) {
            st->pkt1[st->index] = data;
        }
        break;

    case CCE_PACKET3:
        if (st->index < ATI_CCE_MAX_PKT3) {
            st->pkt3[st->index] = data;
        }
        break;

    default:
        break;
    }

    st->index++;
    st->remaining--;

    if (st->remaining) {
        return;
    }

    /* Packet complete. */
    if (type == CCE_PACKET1) {
        unsigned reg0 = (st->hdr & CCE_PACKET1_REG0_MASK) << 2;
        unsigned reg1 = ((st->hdr & CCE_PACKET1_REG1_MASK) >> 11) << 2;

        ati_mm_write(s, reg0, st->pkt1[0], 4);
        ati_mm_write(s, reg1, st->pkt1[1], 4);
    } else if (type == CCE_PACKET3) {
        ati_cce_dispatch_packet3(s, st->hdr, st->pkt3,
                                 MIN(st->index, ATI_CCE_MAX_PKT3));
    }
}

/*
 * PIO FIFO command stream (PM4_FIFO_DATA_EVEN / PM4_FIFO_DATA_ODD).
 *
 * The PM4 PIO modes do not fetch commands from a ring in VRAM; the driver
 * writes the packet stream straight into the chip FIFO, alternating dwords
 * between the even and odd data ports.  Both ports feed one stream, so this
 * is called for either address and simply consumes the next dword.
 */
void ati_cce_fifo_write(ATIVGAState *s, uint32_t data)
{
    ati_cce_stream_dword(s, &s->cce.fifo, data);
}

/*
 * Translate an offset in the card's GART aperture to a guest-physical
 * address.  The GART page table lives in guest memory at PCI_GART_PAGE;
 * each entry is a little-endian 32-bit address of a 4 KiB page (see
 * drm_ati_pcigart_init(), DRM_ATI_GART_PCI: "val = page_base").
 */
static bool ati_gart_translate(ATIVGAState *s, uint32_t off, hwaddr *pa)
{
    uint32_t entry;
    hwaddr pte_addr;

    if (!s->regs.pci_gart_page) {
        return false;
    }
    pte_addr = (hwaddr)s->regs.pci_gart_page + ((off >> 12) * 4);
    pci_dma_read(&s->dev, pte_addr, &entry, sizeof(entry));
    entry = le32_to_cpu(entry);
    if (!entry) {
        return false;
    }
    *pa = ((hwaddr)(entry & ~0xfffULL)) + (off & 0xfff);
    return true;
}

/*
 * Is this video-memory offset inside a buffer the display pipeline owns?
 *
 * Covers the scanned-out surface (CRTC_OFFSET with the CRTC pitch) and the
 * 3D render target (DST_PITCH_OFFSET_C). A texture upload whose destination
 * falls in either is not really establishing a texture there - the address
 * is shared with a display buffer, and reading it back yields frame content
 * rather than texels.
 *
 * Height is not known from the registers alone, so this uses the largest
 * plausible display height. Over-estimating only means a page is left
 * unrecorded and the sampler consults the GART, which is the safe direction;
 * under-estimating would let frame content be sampled as a texture.
 */
static bool ati_addr_in_display_buffer(ATIVGAState *s, uint32_t off)
{
    unsigned i;

    /*
     * The CRTC and 3D-target registers do not describe every display
     * surface: an app can keep its own back/front pair. In one capture the
     * scanout was 0x8000 and the 3D target 0x312000, while Quake 3's front
     * buffer sat at 0x614d00 - named in neither register, yet five texture
     * uploads landed inside it.
     *
     * So the set is learned from traffic instead: ati_2d_note_surface()
     * records any destination the 2D engine blits a full-width span to,
     * which is what presenting a frame looks like. A texture upload into
     * one of those is not establishing a texture.
     */
    for (i = 0; i < ARRAY_SIZE(s->display_surf); i++) {
        uint32_t base = s->display_surf[i].base;
        uint32_t span = s->display_surf[i].span;

        if (span && off >= base && off < base + span) {
            return true;
        }
    }
    return false;
}

/*
 * Remember a blit destination that looks like a display surface: at least
 * 256 pixels wide and covering many lines. Kept in a tiny ring - there are
 * only ever a handful of such buffers.
 */
void ati_2d_note_surface(ATIVGAState *s, uint32_t off, unsigned pitch,
                         unsigned w, unsigned h)
{
    unsigned i;

    if (w < 256 || h < 64 || !pitch) {
        return;
    }
    for (i = 0; i < ARRAY_SIZE(s->display_surf); i++) {
        if (s->display_surf[i].base == off) {
            return;                 /* already known */
        }
    }
    i = s->display_surf_next % ARRAY_SIZE(s->display_surf);
    s->display_surf[i].base = off;
    /*
     * Span the region this blit actually covered, not a worst-case
     * display height. Using pitch * ATI_MAX_DISPLAY_LINES claimed ~3 MiB
     * for a 2560-byte pitch, which swallowed legitimate textures that
     * happen to sit near a framebuffer - gltest2's 8x8 texture among them,
     * leaving it unrecorded so the sampler looked in the GART and found
     * nothing.
     */
    s->display_surf[i].span = pitch * h;
    s->display_surf_next++;
}

static inline unsigned ati_tex_loc_hash(uint32_t page)
{
    return (page >> 12) & (ATI_TEX_LOC_CACHE_SIZE - 1);
}

static void ati_tex_loc_record(ATIVGAState *s, uint32_t page, bool in_gart)
{
    ATITexLoc *e = &s->tex_loc_cache[ati_tex_loc_hash(page)];

    e->page = page;
    e->valid = true;
    e->in_gart = in_gart;
}

/*
 * Returns true and sets *in_gart if this page's actual write location is
 * known; false if nothing has ever recorded a write there, in which case
 * the caller falls back to the GART-first heuristic.
 */
static bool ati_tex_loc_lookup(ATIVGAState *s, uint32_t page, bool *in_gart)
{
    ATITexLoc *e = &s->tex_loc_cache[ati_tex_loc_hash(page)];

    if (!e->valid || e->page != page) {
        return false;
    }
    *in_gart = e->in_gart;
    return true;
}

/*
 * Execute an indirect command buffer.  Called when the guest writes
 * PM4_IW_INDSIZE, which on real hardware starts the fetch.  The buffer sits
 * at PM4_IW_INDOFF within the GART aperture and holds "dwords" dwords of the
 * ordinary packet stream.
 *
 * It gets a fresh decoder state of its own: this runs from inside the FIFO
 * decoder (the INDSIZE write is a packet0 payload dword), so reusing the
 * FIFO's state would clobber the packet still being consumed there.
 *
 * The buffer is walked one 4 KiB page at a time because consecutive GART
 * pages need not be contiguous in guest memory.
 */
void ati_cce_exec_indirect(ATIVGAState *s, uint32_t dwords)
{
    /*
     * Heap, not stack: ATICCEStream now carries a 16384-dword packet3
     * buffer (64 KiB), and this function can recurse up to
     * ATI_CCE_MAX_IND_DEPTH levels, which would overflow the stack.
     */
    ATICCEStream *st;
    uint32_t off = s->cce.iw_indoff;
    uint32_t done = 0;

    if (!dwords) {
        return;
    }
    if (s->cce.ind_depth >= ATI_CCE_MAX_IND_DEPTH) {
        qemu_log_mask(LOG_GUEST_ERROR,
            "ati_cce: indirect buffers nested deeper than %d, not following\n",
            ATI_CCE_MAX_IND_DEPTH);
        return;
    }
    if (dwords > CCE_MAX_INDIRECT_DWORDS) {
        /*
         * Skip, don't truncate. A capture showed submissions of 67600 and
         * 262538 dwords - over a megabyte - among 2959 that exceeded 1000,
         * while genuine buffers are almost all 2 to 96 dwords. Truncating
         * such a value meant walking 64K dwords of unrelated GART memory
         * and dispatching it as commands, which is where the phantom
         * packet3 opcodes (0xff, 0x8f, 0x03) came from and a plausible
         * route to the corrupted driver state behind the ATIRage128
         * kernel panic. An implausible size means the submission itself
         * is not trustworthy, so execute none of it.
         */
        /*
         * Plain qemu_log, not LOG_GUEST_ERROR: that mask only prints with
         * -d guest_errors, so this diagnostic was being discarded on a
         * command line carrying only -d unimp, making it look like the
         * check never fired.
         */
        qemu_log("ati_cce: indirect buffer of %u dwords exceeds %u, "
                 "skipping the submission rather than executing garbage\n",
                 dwords, (unsigned)CCE_MAX_INDIRECT_DWORDS);
        return;
    }

    st = g_new0(ATICCEStream, 1);

    s->cce.ind_depth++;
    while (done < dwords) {
        uint32_t buf[1024];
        /*
         * Bound the chunk by the buffer as well as by the page and the
         * remaining count. The page-boundary term alone can reach exactly
         * 1024 dwords, and nothing else guaranteed chunk stayed within
         * buf[] - a DMA read of more than that would smash the stack.
         */
        uint32_t chunk = MIN(dwords - done,
                             (4096 - (off & 0xfff)) / 4);
        hwaddr pa;
        uint32_t i;

        chunk = MIN(chunk, ARRAY_SIZE(buf));

        if (!chunk) {
            break;
        }
        if (!ati_gart_translate(s, off, &pa)) {
            qemu_log_mask(LOG_GUEST_ERROR,
                "ati_cce: indirect buffer at GART offset 0x%x is not mapped "
                "(PCI_GART_PAGE=0x%x) - %u dwords dropped\n",
                off, s->regs.pci_gart_page, dwords - done);
            break;
        }
        pci_dma_read(&s->dev, pa, buf, chunk * 4);
        for (i = 0; i < chunk; i++) {
            ati_cce_stream_dword(s, st, le32_to_cpu(buf[i]));
        }
        done += chunk;
        off += chunk * 4;
    }
    s->cce.ind_depth--;
    g_free(st);
}
