/*
 * QEMU ATI SVGA emulation
 * 2D engine functions
 *
 * Copyright (c) 2019 BALATON Zoltan
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 */

#include "qemu/osdep.h"
#include "ati_int.h"
#include "ati_regs.h"
#include "qemu/log.h"
#include "qemu/bswap.h"
#include "ui/pixel_ops.h"
#include "ui/console.h"

/*
 * NOTE:
 * This is 2D _acceleration_ and supposed to be fast. Therefore, don't try to
 * reinvent the wheel (unlikely to get better with a naive implementation than
 * existing libraries) and avoid (poorly) reimplementing gfx primitives.
 * That is unnecessary and would become a performance problem. Instead, try to
 * map to and reuse existing optimised facilities (e.g. pixman) wherever
 * possible.
 */

static int ati_bpp_from_datatype(ATIVGAState *s)
{
    switch (s->regs.dp_datatype & 0xf) {
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
        qemu_log_mask(LOG_UNIMP, "Unknown dst datatype %d\n",
                      s->regs.dp_datatype & 0xf);
        return 0;
    }
}

/*
 * src_offset/src_pitch and dst_offset/dst_pitch are always the values to
 * use: ati_mm_write()'s DP_GUI_MASTER_CNTL handler already substitutes
 * default_offset/default_pitch into whichever pair the control word says
 * is not explicitly supplied.
 *
 * The old DEFAULT_CNTL macro tested GMC_DST_PITCH_OFFSET_CNTL and applied
 * the result to the source as well, so a packet that supplies the source
 * but not the destination - which is what Mac OS X sends for a window
 * drag - read the source through the default pitch instead of the one it
 * had just programmed, shearing every copied row.
 */

/*
 * Mark the rows a blit just touched as dirty so the display picks them up.
 *
 * The previous version of this used vbe_start_addr as a byte offset (it is
 * held in dwords), added dst_offset on top of it even though dst_offset is
 * already absolute, and used the *display* stride rather than the blit's
 * own destination stride. The result was that the wrong scanlines got
 * invalidated: parts of each blit were refreshed and parts kept stale
 * content, which showed up as horizontal streaking through freshly drawn
 * areas and as regions that never updated at all.
 *
 * dst_bits already points at the destination surface inside video memory,
 * so the offset to mark is simply its distance from vram_ptr, and the rows
 * are dst_pitch bytes apart.
 */
/*
 * Blit instrumentation.
 *
 * Logs each SRCCOPY through qemu_log(), so it lands in the -D file next to
 * the MMIO trace. Detailed lines are capped so a long session does not bury
 * the log; after the cap only the periodic summary continues, which is
 * enough to tell whether overlapping copies are being seen at all and which
 * copy path they take.
 *
 * "ovl" is whether the source and destination rectangles intersect (a window
 * drag is the usual source of those), "bwd" whether the rows are walked
 * bottom to top to avoid clobbering unread source data, and "path" which
 * implementation ran.
 */
static void ati_2d_log_blt(ATIVGAState *s,
                           unsigned src_x, unsigned src_y,
                           unsigned dst_x, unsigned dst_y,
                           unsigned w, unsigned h,
                           unsigned src_pitch, unsigned dst_pitch,
                           unsigned src_off, unsigned dst_off,
                           bool overlap, bool backwards, const char *path)
{
    static unsigned long total, n_overlap, n_backwards, n_detail;

    total++;
    if (overlap) {
        n_overlap++;
    }
    if (backwards) {
        n_backwards++;
    }

    if (n_detail < 4000) {
        n_detail++;
        qemu_log("ati_2d: blt src(%u,%u) -> dst(%u,%u) %ux%u "
                 "spitch=%u dpitch=%u soff=0x%x doff=0x%x "
                 "ovl=%d bwd=%d path=%s\n",
                 src_x, src_y, dst_x, dst_y, w, h,
                 src_pitch, dst_pitch, src_off, dst_off,
                 overlap, backwards, path);
    }
    if (!(total % 200)) {
        qemu_log("ati_2d: %lu blits, %lu overlapping, %lu reversed\n",
                 total, n_overlap, n_backwards);
    }

    /*
     * Both endpoints of a large blit are candidate display surfaces: a
     * frame present reads one and writes the other. Recording them lets
     * the texture-upload path recognise an address that belongs to a
     * framebuffer rather than to a texture.
     */
    /*
     * Only the DESTINATION indicates a display surface. Recording sources
     * too was wrong: an app blits its rendered frame OUT of a buffer, and
     * textures get read the same way - 82 blits in one capture read from
     * 0x494100, which made the texture at 0x494000 look like framebuffer
     * memory. Every texture was then pushed to the GART, where nothing had
     * written it, and gltest2 and glxgears both lost their textures.
     */
    ati_2d_note_surface(s, dst_off, dst_pitch, w, h);
}

static void ati_2d_mark_dirty(ATIVGAState *s, uint8_t *dst_bits,
                              unsigned dst_y, unsigned dst_pitch)
{
    ram_addr_t offs, len;

    if (!dst_pitch || dst_bits < s->vga.vram_ptr) {
        return;
    }
    offs = (dst_bits - s->vga.vram_ptr) + (ram_addr_t)dst_y * dst_pitch;
    len = (ram_addr_t)s->regs.dst_height * dst_pitch;

    if (offs >= s->vga.vram_size) {
        return;
    }
    len = MIN(len, s->vga.vram_size - offs);
    memory_region_set_dirty(&s->vga.vram, offs, len);
}

void ati_2d_blt(ATIVGAState *s)
{
    /* FIXME it is probably more complex than this and may need to be */
    /* rewritten but for now as a start just to get some output: */
    /*
     * Only the debug print below uses this, and DPRINTF compiles to nothing
     * unless DEBUG_ATI is defined - hence G_GNUC_UNUSED rather than an
     * #ifdef around both the declaration and every use.
     */
    DisplaySurface *ds G_GNUC_UNUSED = qemu_console_surface(s->vga.con);
    DPRINTF("%p %u ds: %p %d %d rop: %x\n", s->vga.vram_ptr,
            s->vga.vbe_start_addr, surface_data(ds), surface_stride(ds),
            surface_bits_per_pixel(ds),
            (s->regs.dp_mix & GMC_ROP3_MASK) >> 16);
    unsigned dst_x = (s->regs.dp_cntl & DST_X_LEFT_TO_RIGHT ?
                      s->regs.dst_x : s->regs.dst_x + 1 - s->regs.dst_width);
    unsigned dst_y = (s->regs.dp_cntl & DST_Y_TOP_TO_BOTTOM ?
                      s->regs.dst_y : s->regs.dst_y + 1 - s->regs.dst_height);
    int bpp = ati_bpp_from_datatype(s);
    unsigned int dst_pitch_bytes;
    if (!bpp) {
        qemu_log_mask(LOG_GUEST_ERROR, "Invalid bpp\n");
        return;
    }
    int dst_stride = s->regs.dst_pitch;
    if (!dst_stride) {
        qemu_log_mask(LOG_GUEST_ERROR, "Zero dest pitch\n");
        return;
    }
    uint8_t *dst_bits = s->vga.vram_ptr + s->regs.dst_offset;

    if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
        /*
         * Do NOT add crtc_offset here.
         *
         * DST_PITCH_OFFSET (and its DST_PITCH_OFFSET_C context alias) carry
         * an absolute offset into video memory, not one relative to the
         * scanout base: an accelerated driver targeting the visible screen
         * programs the framebuffer's own address, which is the same value
         * CRTC_OFFSET holds. Adding crtc_offset on top double-counted it,
         * and worse, pushed blits aimed at offscreen scratch pixmaps (which
         * sit below the framebuffer) up into the visible area, shredding
         * the screen with unrelated drawing.
         *
         * dst_pitch is in 64-byte units in the top 10 bits of the register;
         * as decoded in ati_mm_write it ends up in 8-pixel units, so
         * multiplying by bpp yields the stride in bytes.
         */
        dst_stride *= bpp;
    }
    uint8_t *end = s->vga.vram_ptr + s->vga.vram_size;
    if (dst_x > 0x3fff || dst_y > 0x3fff || dst_bits >= end
        || dst_bits + dst_x
         + (dst_y + s->regs.dst_height) * dst_stride >= end) {
        qemu_log_mask(LOG_UNIMP, "blt outside vram not implemented\n");
        return;
    }
    DPRINTF("%d %d %d, %d %d %d, (%d,%d) -> (%d,%d) %dx%d %c %c\n",
            s->regs.src_offset, s->regs.dst_offset, s->regs.default_offset,
            s->regs.src_pitch, s->regs.dst_pitch, s->regs.default_pitch,
            s->regs.src_x, s->regs.src_y, dst_x, dst_y,
            s->regs.dst_width, s->regs.dst_height,
            (s->regs.dp_cntl & DST_X_LEFT_TO_RIGHT ? '>' : '<'),
            (s->regs.dp_cntl & DST_Y_TOP_TO_BOTTOM ? 'v' : '^'));
    switch (s->regs.dp_mix & GMC_ROP3_MASK) {
    case ROP3_SRCCOPY:
    {
        bool fallback = false;
        bool overlap, backwards;
        unsigned src_x = (s->regs.dp_cntl & DST_X_LEFT_TO_RIGHT ?
                       s->regs.src_x : s->regs.src_x + 1 - s->regs.dst_width);
        unsigned src_y = (s->regs.dp_cntl & DST_Y_TOP_TO_BOTTOM ?
                       s->regs.src_y : s->regs.src_y + 1 - s->regs.dst_height);
        int src_stride = s->regs.src_pitch;
        if (!src_stride) {
            qemu_log_mask(LOG_GUEST_ERROR, "Zero source pitch\n");
            return;
        }
        uint8_t *src_bits = s->vga.vram_ptr + s->regs.src_offset;

        if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
            /* Absolute offset, as for the destination above. */
            src_stride *= bpp;
        }
        if (src_x > 0x3fff || src_y > 0x3fff || src_bits >= end
            || src_bits + src_x
             + (src_y + s->regs.dst_height) * src_stride >= end) {
            qemu_log_mask(LOG_UNIMP, "blt outside vram not implemented\n");
            return;
        }

        /*
         * Does this blit read and write the same pixels?
         *
         * Dragging a window is a copy within the framebuffer, so the source
         * and destination rectangles overlap. pixman_blt() gives no ordering
         * guarantee in that case and smears the image along the copy
         * direction. The old code sidestepped it by using the temporary
         * buffer whenever DP_CNTL asked for a reversed blit, but that is not
         * the same question - a driver that only ever programs the forward
         * direction (as Mac OS X's does) still issues overlapping copies.
         *
         * Compare the actual byte extents of the two rectangles instead.
         * If they overlap, skip pixman and use the memmove loop below, which
         * handles overlap within a row, walking the rows in whichever order
         * keeps us from overwriting source data we have not read yet.
         */
        {
            unsigned int bypp = bpp / 8;
            uint8_t *src_start = src_bits + (size_t)src_y * src_stride
                                          + (size_t)src_x * bypp;
            uint8_t *dst_start = dst_bits + (size_t)dst_y * dst_stride
                                          + (size_t)dst_x * bypp;
            size_t span = (size_t)(s->regs.dst_height - 1) * src_stride
                          + (size_t)s->regs.dst_width * bypp;
            size_t dspan = (size_t)(s->regs.dst_height - 1) * dst_stride
                           + (size_t)s->regs.dst_width * bypp;

            overlap = (src_start < dst_start + dspan) &&
                      (dst_start < src_start + span);
            backwards = overlap && (dst_start > src_start);
        }

        dst_pitch_bytes = dst_stride;
        src_stride /= sizeof(uint32_t);
        dst_stride /= sizeof(uint32_t);
        DPRINTF("pixman_blt(%p, %p, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d)\n",
                src_bits, dst_bits, src_stride, dst_stride, bpp, bpp,
                src_x, src_y, dst_x, dst_y,
                s->regs.dst_width, s->regs.dst_height);
#ifdef CONFIG_PIXMAN
        if (!overlap && (s->use_pixman & BIT(1)) &&
            s->regs.dp_cntl & DST_X_LEFT_TO_RIGHT &&
            s->regs.dp_cntl & DST_Y_TOP_TO_BOTTOM) {
            fallback = !pixman_blt((uint32_t *)src_bits, (uint32_t *)dst_bits,
                                   src_stride, dst_stride, bpp, bpp,
                                   src_x, src_y, dst_x, dst_y,
                                   s->regs.dst_width, s->regs.dst_height);
        } else if (!overlap && (s->use_pixman & BIT(1))) {
            /* FIXME: We only really need a temporary if src and dst overlap */
            int llb = s->regs.dst_width * (bpp / 8);
            int tmp_stride = DIV_ROUND_UP(llb, sizeof(uint32_t));
            uint32_t *tmp = g_malloc(tmp_stride * sizeof(uint32_t) *
                                     s->regs.dst_height);
            fallback = !pixman_blt((uint32_t *)src_bits, tmp,
                                   src_stride, tmp_stride, bpp, bpp,
                                   src_x, src_y, 0, 0,
                                   s->regs.dst_width, s->regs.dst_height);
            if (!fallback) {
                fallback = !pixman_blt(tmp, (uint32_t *)dst_bits,
                                       tmp_stride, dst_stride, bpp, bpp,
                                       0, 0, dst_x, dst_y,
                                       s->regs.dst_width, s->regs.dst_height);
            }
            g_free(tmp);
        } else
#endif
        {
            fallback = true;
        }
        if (fallback) {
            unsigned int y, i, j, bypp = bpp / 8;
            unsigned int src_pitch = src_stride * sizeof(uint32_t);
            unsigned int dst_pitch = dst_stride * sizeof(uint32_t);

            for (y = 0; y < s->regs.dst_height; y++) {
                i = dst_x * bypp;
                j = src_x * bypp;
                /*
                 * Walk top to bottom normally. When the rectangles overlap
                 * and the destination sits later in memory than the source,
                 * go bottom to top instead so each row is read before a
                 * later row's write can clobber it.
                 */
                if (!backwards &&
                    (overlap || (s->regs.dp_cntl & DST_Y_TOP_TO_BOTTOM))) {
                    i += (dst_y + y) * dst_pitch;
                    j += (src_y + y) * src_pitch;
                } else {
                    i += (dst_y + s->regs.dst_height - 1 - y) * dst_pitch;
                    j += (src_y + s->regs.dst_height - 1 - y) * src_pitch;
                }
                memmove(&dst_bits[i], &src_bits[j], s->regs.dst_width * bypp);
            }
        }
        ati_2d_log_blt(s, src_x, src_y, dst_x, dst_y,
                       s->regs.dst_width, s->regs.dst_height,
                       src_stride * (unsigned)sizeof(uint32_t),
                       dst_pitch_bytes,
                       (unsigned)(src_bits - s->vga.vram_ptr),
                       (unsigned)(dst_bits - s->vga.vram_ptr),
                       overlap, backwards,
                       fallback ? "memmove" : "pixman");
        ati_2d_mark_dirty(s, dst_bits, dst_y, dst_pitch_bytes);
        s->regs.dst_x = (s->regs.dp_cntl & DST_X_LEFT_TO_RIGHT ?
                         dst_x + s->regs.dst_width : dst_x);
        s->regs.dst_y = (s->regs.dp_cntl & DST_Y_TOP_TO_BOTTOM ?
                         dst_y + s->regs.dst_height : dst_y);
        break;
    }
    case ROP3_PATCOPY:
    case ROP3_BLACKNESS:
    case ROP3_WHITENESS:
    {
        uint32_t filler = 0;

        switch (s->regs.dp_mix & GMC_ROP3_MASK) {
        case ROP3_PATCOPY:
            filler = s->regs.dp_brush_frgd_clr;
            break;
        case ROP3_BLACKNESS:
            filler = 0xffUL << 24 | rgb_to_pixel32(s->vga.palette[0],
                     s->vga.palette[1], s->vga.palette[2]);
            break;
        case ROP3_WHITENESS:
            filler = 0xffUL << 24 | rgb_to_pixel32(s->vga.palette[3],
                     s->vga.palette[4], s->vga.palette[5]);
            break;
        }

        /*
         * pixman_fill() and the fallback below both store the fill colour
         * in HOST byte order, which is a different thing from the guest's.
         * The framebuffer here is big-endian ARGB, so on a little-endian
         * host an opaque black 0xff000000 lands as bytes 00 00 00 ff and
         * is read back as alpha 0, blue 255 - a blue screen instead of a
         * black one. Pre-swap so the bytes reach memory the right way
         * round, matching what ati_3d.c already does via stl_be_p.
         */
        if (s->vga.big_endian_fb != (HOST_BIG_ENDIAN != 0)) {
            filler = bswap32(filler);
        }

        dst_pitch_bytes = dst_stride;
        dst_stride /= sizeof(uint32_t);
        DPRINTF("pixman_fill(%p, %d, %d, %d, %d, %d, %d, %x)\n",
                dst_bits, dst_stride, bpp, dst_x, dst_y,
                s->regs.dst_width, s->regs.dst_height, filler);
#ifdef CONFIG_PIXMAN
        if (!(s->use_pixman & BIT(0)) ||
            !pixman_fill((uint32_t *)dst_bits, dst_stride, bpp, dst_x, dst_y,
                    s->regs.dst_width, s->regs.dst_height, filler))
#endif
        {
            /* fallback when pixman failed or we don't want to call it */
            unsigned int x, y, i, bypp = bpp / 8;
            unsigned int dst_pitch = dst_stride * sizeof(uint32_t);
            for (y = 0; y < s->regs.dst_height; y++) {
                i = dst_x * bypp + (dst_y + y) * dst_pitch;
                for (x = 0; x < s->regs.dst_width; x++, i += bypp) {
                    stn_he_p(&dst_bits[i], bypp, filler);
                }
            }
        }
        ati_2d_mark_dirty(s, dst_bits, dst_y, dst_pitch_bytes);
        s->regs.dst_y = (s->regs.dp_cntl & DST_Y_TOP_TO_BOTTOM ?
                         dst_y + s->regs.dst_height : dst_y);
        break;
    }
    default:
        qemu_log_mask(LOG_UNIMP, "Unimplemented ati_2d blt op %x\n",
                      (s->regs.dp_mix & GMC_ROP3_MASK) >> 16);
    }
}
