/*
 * QEMU ATI SVGA emulation
 *
 * Copyright (c) 2019 BALATON Zoltan
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 */

#ifndef ATI_INT_H
#define ATI_INT_H

#include "qemu/timer.h"
#include "qemu/units.h"
#include "hw/pci/pci_device.h"
#include "hw/i2c/bitbang_i2c.h"
#include "hw/display/i2c-ddc.h"
#include "vga_int.h"
#include "qom/object.h"

/*#define DEBUG_ATI*/

#ifdef DEBUG_ATI
#define DPRINTF(fmt, ...) printf("%s: " fmt, __func__, ## __VA_ARGS__)
#else
#define DPRINTF(fmt, ...) do {} while (0)
#endif

#define PCI_VENDOR_ID_ATI 0x1002
/* Rage128 Pro GL */
#define PCI_DEVICE_ID_ATI_RAGE128_PF 0x5046
/* Radeon RV100 (VE) */
#define PCI_DEVICE_ID_ATI_RADEON_QY 0x5159

#define ATI_RAGE128_LINEAR_APER_SIZE (64 * MiB)
#define ATI_R100_LINEAR_APER_SIZE (128 * MiB)
#define ATI_HOST_DATA_ACC_BITS 128

#define TYPE_ATI_VGA "ati-vga"
OBJECT_DECLARE_SIMPLE_TYPE(ATIVGAState, ATI_VGA)

typedef struct ATIVGARegs {
    uint32_t mm_index;
    uint32_t bios_scratch[8];
    uint32_t gen_int_cntl;
    uint32_t gen_int_status;
    uint32_t crtc_gen_cntl;
    uint32_t crtc_ext_cntl;
    uint32_t dac_cntl;
    uint32_t gpio_vga_ddc;
    uint32_t gpio_dvi_ddc;
    uint32_t gpio_monid;
    uint32_t config_cntl;
    uint32_t mem_cntl;
    /* 3D texture unit (the _C context block) */
    uint32_t tex_cntl;          /* 0x1c9c TEX_CNTL_C */
    uint32_t prim_tex_cntl;     /* 0x1cb0 PRIM_TEX_CNTL_C */
    uint32_t prim_tex_combine;  /* 0x1cb4 PRIM_TEXTURE_COMBINE_CNTL_C */
    uint32_t tex_size_pitch;    /* 0x1cb8 TEX_SIZE_PITCH_C */
    uint32_t prim_tex_offset;   /* 0x1cbc PRIM_TEX_0_OFFSET_C */
    uint32_t misc_3d_state;     /* 0x1ca0 MISC_3D_STATE_CNTL_REG */
    uint32_t setup_cntl;        /* 0x1bc4 SETUP_CNTL - winding/ST-mode/etc */
    uint32_t vc_fpu_setup;      /* 0x071c PM4_VC_FPU_SETUP - front dir, cull */
    /*
     * The 3D context has its OWN destination pitch/offset (0x1c80,
     * DST_PITCH_OFFSET_C) separate from the 2D engine's (0x142c,
     * DST_PITCH_OFFSET). They are programmed independently and to
     * different values - in one Tux Racer capture 0x1c80 was written
     * 10704 times with a 2560-byte pitch at 0x312000 while 0x142c was
     * written 2353 times with a 4096-byte pitch at 0x8000. Sharing one
     * pair of globals meant every 2D blit clobbered the 3D render target
     * and every 3D draw clobbered the blit destination, so whichever
     * engine ran last decided where the other one wrote.
     */
    uint32_t dst_offset_3d;
    uint32_t dst_pitch_3d;
    uint32_t dst_tile_3d;

    uint32_t z_offset;           /* 0x1c90 Z_OFFSET_C */
    uint32_t z_pitch;            /* 0x1c94 Z_PITCH_C */
    uint32_t z_sten_cntl;        /* 0x1c98 Z_STEN_CNTL_C */

    uint32_t sec_tex_cntl;       /* 0x1d00 SEC_TEX_CNTL_C */
    uint32_t sec_tex_combine;    /* 0x1d04 SEC_TEX_COMBINE_CNTL_C */
    uint32_t sec_tex_offset;     /* 0x1d08 SEC_TEX_0_OFFSET_C */
    uint32_t palette[256];
    uint32_t crtc_h_total_disp;
    uint32_t crtc_h_sync_strt_wid;
    uint32_t crtc_v_total_disp;
    uint32_t crtc_v_sync_strt_wid;
    uint32_t crtc_offset;
    uint32_t crtc_offset_cntl;
    uint32_t crtc_pitch;
    uint32_t cur_offset;
    uint32_t cur_hv_pos;
    uint32_t cur_hv_offs;
    uint32_t cur_color0;
    uint32_t cur_color1;
    uint32_t dst_offset;
    uint32_t dst_pitch;
    uint32_t dst_tile;
    uint32_t dst_width;
    uint32_t dst_height;
    uint32_t src_offset;
    uint32_t src_pitch;
    uint32_t src_tile;
    uint32_t src_x;
    uint32_t src_y;
    uint32_t dst_x;
    uint32_t dst_y;
    uint32_t dp_gui_master_cntl;
    uint32_t dp_brush_bkgd_clr;
    uint32_t dp_brush_frgd_clr;
    uint32_t dp_src_frgd_clr;
    uint32_t dp_src_bkgd_clr;
    uint32_t gui_scratch[6]; /* GUI_SCRATCH_REG0..5, 0x15e0-0x15f4 */
    uint32_t pci_gart_page;  /* guest-phys address of the GART page table */
    uint32_t mc_agp_location;
    uint32_t mc_fb_location;
    uint16_t sc_top;
    uint16_t sc_left;
    uint16_t sc_bottom;
    uint16_t sc_right;
    uint16_t src_sc_bottom;
    uint16_t src_sc_right;
    uint32_t dp_cntl;
    uint32_t dp_datatype;
    uint32_t dp_mix;
    uint32_t dp_write_mask;
    uint32_t default_offset;
    uint32_t default_pitch;
    uint16_t default_sc_bottom;
    uint16_t default_sc_right;
    uint32_t default_tile;
} ATIVGARegs;

/*
 * CCE (Concurrent Command Engine) state. Rage128 hardware calls this the
 * "PM4" engine in its own register names (see PM4_BUFFER_* etc in
 * ati_regs.h) - "CCE" is ATI's driver/software-stack name for the same
 * unit. This ring lives in the card's own VRAM and carries a packet
 * stream in the type0/1/2/3 format shared with the later Radeon PM4
 * engine (see ati_3d.c for packet layout, sourced from the open r128
 * DRM/DRI driver headers).
 *
 * NOTE: this only supports the buffer-in-VRAM (PIO-style) operating
 * mode. It does not yet implement the bus-mastering indirect-buffer
 * mode (PM4_*BM* values of PM4_BUFFER_CNTL) where ring entries are
 * pointers to command buffers in system/AGP memory fetched via DMA -
 * that is the mode real OS drivers prefer for performance and is not
 * wired up here yet.
 */
/* Largest packet3 payload buffered from a command stream before dispatch. */
/*
 * Largest packet3 payload buffered from a command stream before dispatch.
 *
 * This was 64, which silently truncated real geometry: the CCE packet count
 * field is 14 bits, so a payload can legitimately reach 16384 dwords, and a
 * single cube face batch is 24 vertices x 10 dwords + 2 = 242. With a
 * 64-dword buffer the vertex count was clamped to (64-2)/10 = 6 - exactly
 * one quad - so every 3D draw lost everything past its first face. Sized to
 * the hardware maximum so no legal packet is cut short.
 */
#define ATI_CCE_MAX_PKT3 16384

/* How deep an indirect buffer may nest before we stop following it. */
#define ATI_CCE_MAX_IND_DEPTH 4

/* Decoder state for a single packet0/1/2/3 command stream. */
typedef struct ATICCEStream {
    uint32_t hdr;            /* header of the packet being consumed */
    unsigned remaining;      /* payload dwords still expected, 0 = want hdr */
    unsigned index;          /* index of the next payload dword */
    unsigned reg;            /* packet0 target register, byte offset */
    bool one_reg;            /* packet0 ONE_REG_WR: don't advance the reg */
    uint32_t pkt1[2];        /* packet1 payload */
    uint32_t pkt3[ATI_CCE_MAX_PKT3];
    uint32_t stream_pos;     /* dwords consumed - for locating a desync */
} ATICCEStream;

typedef struct ATICCEState {
    uint32_t buffer_cntl;   /* raw PM4_BUFFER_CNTL, mode + size bits */
    uint32_t buffer_offset; /* PM4_BUFFER_OFFSET: byte offset into VRAM */
    uint32_t rptr;          /* PM4_BUFFER_DL_RPTR: dword index into ring */
    uint32_t wptr;          /* PM4_BUFFER_DL_WPTR: dword index into ring */
    /* Setup-engine state latched by packet0 writes, consumed by 3D pkts */
    uint32_t vc_format;     /* PM4_VC_FORMAT: vertex format flags */
    uint32_t vc_cntl;       /* PM4_VC_CNTL: prim type/walk/count */

    /*
     * Decoder state for one packet stream.  There is one instance for the
     * PIO FIFO and a separate one per indirect buffer, because an indirect
     * buffer is fetched from the middle of a FIFO packet (the write to
     * PM4_IW_INDSIZE that triggers it is itself a packet0 payload dword).
     * Sharing a single state across both would clobber the outer packet's
     * header and counters and desync the stream.
     *
     * Decoded as a stream rather than buffered whole: packet0 register
     * writes are dispatched as each payload dword arrives, so a burst of
     * any length works without a large buffer.  Only packet1 (2 dwords)
     * and packet3 (capped) need to be held until complete.
     */
    ATICCEStream fifo;       /* PM4_FIFO_DATA_EVEN / _ODD stream */

    /*
     * Indirect buffer submission (PM4_IW_INDOFF / PM4_IW_INDSIZE).
     *
     * In the "INDBM" PM4 modes the FIFO does not carry the drawing
     * commands themselves - it carries a pointer to a command buffer in
     * system memory.  The driver writes the buffer's offset within the
     * card's GART aperture to PM4_IW_INDOFF and its length in dwords to
     * PM4_IW_INDSIZE; the write to INDSIZE is what kicks off the fetch.
     *
     * Addresses are translated through the PCI GART: PCI_GART_PAGE holds
     * the guest-physical address of a page table whose entries are plain
     * little-endian 4 KiB page addresses (ati_pcigart.c, DRM_ATI_GART_PCI).
     */
    uint32_t iw_indoff;      /* PM4_IW_INDOFF: byte offset into the GART */
    unsigned ind_depth;      /* nesting guard for indirect-within-indirect */
} ATICCEState;

/*
 * Records where each texture upload actually landed - VRAM or GART - keyed
 * by the destination's 4 KiB page. The sampler consults this instead of
 * re-deciding independently: GART translation succeeding is not proof a
 * texture lives there, since the page table covers the whole aperture and
 * will "translate" almost any offset to *some* physical address, whether or
 * not that address holds this texture's data. A texture genuinely uploaded
 * to VRAM (which happens when a host-data blit's destination fails GART
 * translation) was being shadowed by an unrelated, coincidentally-valid
 * GART entry at the same virtual offset. Tying the read to the write closes
 * that gap.
 */
#define ATI_TEX_LOC_CACHE_SIZE 256

typedef struct ATITexLoc {
    uint32_t page;
    bool valid;
    bool in_gart;
} ATITexLoc;

typedef struct ATIHostDataState {
    bool active;
    uint32_t row;
    uint32_t col;
    uint32_t next;
    uint32_t acc[4];
} ATIHostDataState;

struct ATIVGAState {
    PCIDevice dev;
    VGACommonState vga;
    char *model;
    uint16_t dev_id;
    uint8_t mode;
    uint8_t use_pixman;
    bool cursor_guest_mode;
    uint16_t cursor_size;
    uint32_t cursor_offset;
    QEMUCursor *cursor;
    QEMUTimer vblank_timer;
    bitbang_i2c_interface bbi2c;
    I2CDDCState i2cddc;
    uint64_t linear_aper_sz;
    MemoryRegion linear_aper;
    MemoryRegion linear_aper1;
    MemoryRegion io;
    MemoryRegion mm;
    ATIVGARegs regs;
    ATIHostDataState host_data;
    ATICCEState cce;
    ATITexLoc tex_loc_cache[ATI_TEX_LOC_CACHE_SIZE];

    /*
     * Video-memory regions the 2D engine treats as display surfaces,
     * learned from blit traffic. A texture upload landing in one of these
     * is sharing an address with a framebuffer, so it must not be recorded
     * as a texture location - reading it back returns presented frame
     * content instead of texels.
     */
    struct {
        uint32_t base;
        uint32_t span;
    } display_surf[8];
    unsigned display_surf_next;
};

const char *ati_reg_name(int num);

void ati_2d_blt(ATIVGAState *s);
bool ati_host_data_flush(ATIVGAState *s);

/* Exposed so ati_3d.c can dispatch decoded packet0/1 register writes
 * through the same logic MMIO writes use, instead of duplicating it. */
void ati_mm_write(void *opaque, hwaddr addr, uint64_t data, unsigned int size);

/* ati_3d.c: CCE ring processing, called when the guest advances
 * PM4_BUFFER_DL_WPTR (i.e. tells the card there are new commands). */
void ati_cce_process(ATIVGAState *s);

/* ati_3d.c: feed one dword written to PM4_FIFO_DATA_EVEN/ODD into the
 * PIO command stream decoder. */
void ati_cce_fifo_write(ATIVGAState *s, uint32_t data);

/* ati_3d.c: fetch and execute an indirect command buffer, triggered by a
 * write to PM4_IW_INDSIZE. */
void ati_cce_exec_indirect(ATIVGAState *s, uint32_t dwords);
void ati_2d_note_surface(ATIVGAState *s, uint32_t off, unsigned pitch,
                         unsigned w, unsigned h);
void ati_host_data_finish(ATIVGAState *s);

#endif /* ATI_INT_H */
