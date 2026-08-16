#!/usr/bin/env python3
"""VCK190 power/temp sampler: reads all INA226 rails via i2c-0 mux 0x74 + hwmon temps.

Rail table from Xilinx system-controller-app board/VCK190.json
(Shunt_Resistor in micro-ohm, Phase_Multiplier scales measured current).
Writes CSV to argv[1] until killed.
"""
import os, sys, fcntl, time, struct

I2C_SLAVE = 0x0703
MUX = 0x74
BUS = "/dev/i2c-0"

# (mux_channel, addr, name, shunt_uohm, phase_mult)
RAILS = [
    (1, 0x40, "VCCINT",      500, 6),
    (1, 0x41, "VCC_SOC",     500, 1),
    (1, 0x42, "VCC_PMC",    5000, 1),
    (1, 0x43, "VCC_RAM",    5000, 1),
    (1, 0x44, "VCCINT_PSLP",5000, 1),
    (1, 0x45, "VCCINT_PSFP",5000, 1),
    (3, 0x40, "VCCAUX",     5000, 1),
    (3, 0x41, "VCCAUX_PMC", 5000, 1),
    (3, 0x45, "VCC_MIO",    5000, 1),
    (3, 0x46, "VCC1V8",     5000, 1),
    (3, 0x47, "VCC3V3",     5000, 1),
    (3, 0x48, "VCC_DDR4",   5000, 1),
    (3, 0x49, "VCC1V1_LP4", 5000, 1),
    (3, 0x4A, "VADJ_FMC",   2000, 1),
    (3, 0x4B, "MGTYAVCC",   2000, 1),
    (3, 0x4C, "MGTYAVTT",   2000, 1),
    (3, 0x4D, "MGTYVCCAUX", 5000, 1),
]

TEMPS = [
    ("versal", "/sys/class/hwmon/hwmon0/temp1_input"),
    ("aie",    "/sys/class/hwmon/hwmon1/temp1_input"),
    ("sysmon_max", "/sys/bus/iio/devices/iio:device0/in_temp162_max_max_input"),
]
# per-channel instantaneous AIE sensors, when the design exposes them
import glob as _glob
for _p in sorted(_glob.glob("/sys/bus/iio/devices/iio:device0/in_temp200_aie-temp-ch*_input"),
                 key=lambda p: int(p.rsplit("ch", 1)[1].split("_")[0])):
    _ch = _p.rsplit("ch", 1)[1].split("_")[0]
    TEMPS.append((f"aie_ch{_ch}", _p))

def rd16(fd, addr, reg):
    fcntl.ioctl(fd, I2C_SLAVE, addr)
    os.write(fd, bytes([reg]))
    return struct.unpack(">H", os.read(fd, 2))[0]

def setmux(fd, mask):
    fcntl.ioctl(fd, I2C_SLAVE, MUX)
    os.write(fd, bytes([mask]))

def read_rail(fd, ch, addr, shunt_uohm, pmult):
    """Returns (bus_V, power_W) or None on integrity failure."""
    for _ in range(3):
        try:
            setmux(fd, 1 << ch)
            if rd16(fd, addr, 0xFF) != 0x2260:   # die ID check
                continue
            vsh_raw = rd16(fd, addr, 0x01)
            if vsh_raw >= 0x8000:
                vsh_raw -= 0x10000
            vbus_raw = rd16(fd, addr, 0x02)
            v = vbus_raw * 1.25e-3
            i = (vsh_raw * 2.5e-6) / (shunt_uohm * 1e-6) * pmult
            return v, v * i
        except OSError:
            time.sleep(0.01)
    return None

def read_temp(path):
    try:
        with open(path) as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return float("nan")

def main(out_path):
    fd = os.open(BUS, os.O_RDWR)
    with open(out_path, "w", buffering=1) as out:
        cols = ["epoch"] + [n for n, _ in TEMPS] + \
               [f"{r[2]}_V" for r in RAILS] + [f"{r[2]}_W" for r in RAILS] + ["total_W"]
        out.write(",".join(cols) + "\n")
        while True:
            t0 = time.time()
            temps = [read_temp(p) for _, p in TEMPS]
            volts, watts = [], []
            for ch, addr, name, sh, pm in RAILS:
                r = read_rail(fd, ch, addr, sh, pm)
                if r is None:
                    volts.append(float("nan")); watts.append(float("nan"))
                else:
                    volts.append(r[0]); watts.append(r[1])
            try:
                setmux(fd, 0)
            except OSError:
                pass
            total = sum(w for w in watts if w == w)
            row = [f"{t0:.2f}"] + [f"{t:.3f}" for t in temps] + \
                  [f"{v:.4f}" for v in volts] + [f"{w:.4f}" for w in watts] + [f"{total:.4f}"]
            out.write(",".join(row) + "\n")
            dt = time.time() - t0
            if dt < 1.0:
                time.sleep(1.0 - dt)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/root/thermal_log.csv")
