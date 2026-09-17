/*
 * PowerMac NewWorld MacIO GPIO emulation
 *
 * Copyright (c) 2016 Benjamin Herrenschmidt
 * Copyright (c) 2018 Mark Cave-Ayland
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "qemu/osdep.h"
#include "hw/qdev-properties.h"
#include "migration/vmstate.h"
#include "hw/misc/macio/macio.h"
#include "hw/misc/macio/gpio.h"
#include "hw/irq.h"
#include "hw/nmi.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "trace.h"

enum MacioGPIORegisterBits {
    OUT_DATA   = 1,
    IN_DATA    = 2,
    OUT_ENABLE = 4,
};

void macio_set_gpio(MacIOGPIOState *s, uint32_t gpio, bool state)
{
    uint8_t new_reg;

    trace_macio_set_gpio(gpio, state);

    if (s->gpio_regs[gpio] & OUT_ENABLE) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "GPIO: Setting GPIO %d while it's an output\n", gpio);
    }

    new_reg = s->gpio_regs[gpio] & ~IN_DATA;
    if (state) {
        new_reg |= IN_DATA;
    }

    if (new_reg == s->gpio_regs[gpio]) {
        return;
    }

    s->gpio_regs[gpio] = new_reg;

    /*
     * Note that we probably need to get access to the MPIC config to
     * decode polarity since qemu always use "raise" regardless.
     *
     * For now, we hard wire known GPIOs
     */

    switch (gpio) {
    case 1:
        /* Level low */
        if (!state) {
            trace_macio_gpio_irq_assert(gpio);
            qemu_irq_raise(s->gpio_extirqs[gpio]);
        } else {
            trace_macio_gpio_irq_deassert(gpio);
            qemu_irq_lower(s->gpio_extirqs[gpio]);
        }
        break;
    case 4:
        /* Active low, CPU1 reset */
        if (!state) {
            trace_macio_gpio_irq_assert(gpio);
            qemu_irq_raise(s->gpio_extirqs[gpio]);
        } else {
            trace_macio_gpio_irq_deassert(gpio);
            qemu_irq_lower(s->gpio_extirqs[gpio]);
        }
        break;
    case 5:
        /* Active low, CPU2 reset */
        if (!state) {
            trace_macio_gpio_irq_assert(gpio);
            qemu_irq_raise(s->gpio_extirqs[gpio]);
        } else {
            trace_macio_gpio_irq_deassert(gpio);
            qemu_irq_lower(s->gpio_extirqs[gpio]);
        }
        break;
    case 6:
        /* Active low, CPU3 reset */
        if (!state) {
            trace_macio_gpio_irq_assert(gpio);
            qemu_irq_raise(s->gpio_extirqs[gpio]);
        } else {
            trace_macio_gpio_irq_deassert(gpio);
            qemu_irq_lower(s->gpio_extirqs[gpio]);
        }
        break;
    case 9:
        /* Edge, triggered by NMI below */
        if (state) {
            trace_macio_gpio_irq_assert(gpio);
            qemu_irq_raise(s->gpio_extirqs[gpio]);
        } else {
            trace_macio_gpio_irq_deassert(gpio);
            qemu_irq_lower(s->gpio_extirqs[gpio]);
        }
        break;

    default:
        qemu_log_mask(LOG_UNIMP, "GPIO: setting unimplemented GPIO %d", gpio);
    }
}

static void macio_cpu_reset_irq(MacIOGPIOState *s, uint32_t gpio, bool state)
{
    /*
     * Fires the CPU-reset qemu_irq line only. Deliberately does NOT touch
     * s->gpio_regs[] - the real register storage for the GPIO address that
     * triggered this is handled separately by the caller, since the
     * compressed irq-line index used here (4/5/6) is not the same as the
     * real GPIO register address (4/15/16) and must never be used to index
     * gpio_regs[] directly.
     */
    if (!state) {
        trace_macio_gpio_irq_assert(gpio);
        qemu_irq_raise(s->gpio_extirqs[gpio]);
    } else {
        trace_macio_gpio_irq_deassert(gpio);
        qemu_irq_lower(s->gpio_extirqs[gpio]);
    }
}

static void macio_gpio_write(void *opaque, hwaddr addr, uint64_t value,
                             unsigned size)
{
    MacIOGPIOState *s = opaque;
    uint8_t ibit;

    trace_macio_gpio_write(addr, value);

    /* Levels regs are read-only */
    if (addr < 8) {
        return;
    }

    addr -= 8;
    if (addr < 36) {
        value &= ~IN_DATA;

        if (value & OUT_ENABLE) {
            ibit = (value & OUT_DATA) << 1;
        } else {
            ibit = s->gpio_regs[addr] & IN_DATA;
        }

        /*
         * Always store the real register at its real address first, for
         * every GPIO - this matches the original (pre-SMP-patch) behavior
         * and must never be skipped, regardless of whether addr also
         * happens to be one of the CPU-reset lines below. macio_set_gpio()
         * is an internal hardware-signal interface (see macio_gpio_nmi())
         * driven by other code, not something every guest MMIO write
         * should invoke - the SMP patch incorrectly called it here.
         */
        s->gpio_regs[addr] = value | ibit;

        if (addr == (KL_GPIO_RESET_CPU1 - KEYLARGO_GPIO_EXTINT_0)) {
            /* CPU1 reset: (0x58+0x04) - 0x58 = 0x04, irq line 4 */
            if (!(value & OUT_ENABLE)) {
                ibit = 1; /* Ensure pulled high unless driven low */
            }
            macio_cpu_reset_irq(s, 4, ibit);
        } else if (addr == (KL_GPIO_RESET_CPU2 - KEYLARGO_GPIO_EXTINT_0)) {
            /* CPU2 reset: (0x58+0x0f) - 0x58 = 0x0F, irq line 5 */
            macio_cpu_reset_irq(s, 5, ibit);
        } else if (addr == (KL_GPIO_RESET_CPU3 - KEYLARGO_GPIO_EXTINT_0)) {
            /* CPU3 reset: (0x58+0x10) - 0x58 = 0x10, irq line 6 */
            macio_cpu_reset_irq(s, 6, ibit);
        }
    }
}

static uint64_t macio_gpio_read(void *opaque, hwaddr addr, unsigned size)
{
    MacIOGPIOState *s = opaque;
    uint64_t val = 0;

    /* Levels regs */
    if (addr < 8) {
        val = s->gpio_levels[addr];
    } else {
        addr -= 8;

        if (addr < 36) {
            val = s->gpio_regs[addr];
        }
    }

    trace_macio_gpio_read(addr, val);
    return val;
}

static const MemoryRegionOps macio_gpio_ops = {
    .read = macio_gpio_read,
    .write = macio_gpio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 1,
    },
};

static void macio_gpio_init(Object *obj)
{
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);
    MacIOGPIOState *s = MACIO_GPIO(obj);
    int i;

    for (i = 0; i < 10; i++) {
        sysbus_init_irq(sbd, &s->gpio_extirqs[i]);
    }

    memory_region_init_io(&s->gpiomem, OBJECT(s), &macio_gpio_ops, obj,
                          "gpio", 0x30);
    sysbus_init_mmio(sbd, &s->gpiomem);
}

static const VMStateDescription vmstate_macio_gpio = {
    .name = "macio_gpio",
    .version_id = 0,
    .minimum_version_id = 0,
    .fields = (const VMStateField[]) {
        VMSTATE_UINT8_ARRAY(gpio_levels, MacIOGPIOState, 8),
        VMSTATE_UINT8_ARRAY(gpio_regs, MacIOGPIOState, 36),
        VMSTATE_END_OF_LIST()
    }
};

static void macio_gpio_reset(DeviceState *dev)
{
    MacIOGPIOState *s = MACIO_GPIO(dev);

    /* GPIO 1 is up by default */
    macio_set_gpio(s, 1, true);
    macio_set_gpio(s, 4, true);
    macio_set_gpio(s, 5, true);
    macio_set_gpio(s, 6, true);
}

static void macio_gpio_nmi(NMIState *n, int cpu_index, Error **errp)
{
    macio_set_gpio(MACIO_GPIO(n), 9, true);
    macio_set_gpio(MACIO_GPIO(n), 9, false);
}

static void macio_gpio_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    NMIClass *nc = NMI_CLASS(oc);

    device_class_set_legacy_reset(dc, macio_gpio_reset);
    dc->vmsd = &vmstate_macio_gpio;
    nc->nmi_monitor_handler = macio_gpio_nmi;
}

static const TypeInfo macio_gpio_init_info = {
    .name          = TYPE_MACIO_GPIO,
    .parent        = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(MacIOGPIOState),
    .instance_init = macio_gpio_init,
    .class_init    = macio_gpio_class_init,
    .interfaces = (InterfaceInfo[]) {
        { TYPE_NMI },
        { }
    },
};

static void macio_gpio_register_types(void)
{
    type_register_static(&macio_gpio_init_info);
}

type_init(macio_gpio_register_types)
