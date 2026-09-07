#!/usr/bin/env python3
"""Read the VCCINT regulator's OWN output telemetry over PMBus (IR35215 at 0x16 on
i2c-0 switch U33 0x75 port 0, per UG1366), instead of the single-phase INA226.

Why: the INA226 at 0x40 senses ONE of the six VCCINT phases and the sampler
multiplies by 6. Under light load the controller sheds phases, so that phase
can carry ~zero or slightly negative current -> the "VCCINT power" reads
anywhere from -1.8 W to +1.8 W for the same image. READ_POUT from the
controller integrates all phases. LINEAR11 decoding per PMBus spec.
Run on the board while the sampler is NOT running: they share switch 0x74 and the
sampler re-points it every second, which corrupts these reads (seen 2026-09-07).
"""
import fcntl, os, struct, sys, time
I2C_SLAVE = 0x0703
MUX, CH, ADDR = 0x74, 0, 0x16   # regulators sit on i2c-0 switch 0x74 channel 0 (probed 2026-09-07)
def dev(a):
    fd = os.open("/dev/i2c-0", os.O_RDWR); fcntl.ioctl(fd, I2C_SLAVE, a); return fd
def lin11(w):
    m = w & 0x7FF; e = (w >> 11) & 0x1F
    if m >= 0x400: m -= 0x800
    if e >= 0x10: e -= 0x20
    return m * 2.0 ** e
def rd_word(fd, cmd):
    os.write(fd, bytes([cmd])); b = os.read(fd, 2); return b[0] | (b[1] << 8)
mux = dev(MUX); os.write(mux, bytes([1 << CH])); os.close(mux)
fd = dev(ADDR)
def page(p): os.write(fd, bytes([0x00, p]))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
print("t,page,vout_V,iout_A,pout_W,temp_C")
for _ in range(n):
    for p in (0, 1):                       # page 0 = VCCINT loop, page 1 = VCC_SOC loop (IR35215 dual-loop)
        page(p)
        vmode = rd_word(fd, 0x20) & 0xFF   # VOUT_MODE
        vraw = rd_word(fd, 0x8B); iout = lin11(rd_word(fd, 0x8C)); pout = lin11(rd_word(fd, 0x96)); tmp = lin11(rd_word(fd, 0x8D))
        exp = vmode & 0x1F; exp = exp - 32 if exp >= 16 else exp
        vout = vraw * 2.0 ** exp
        print(f"{time.time():.2f},{p},{vout:.4f},{iout:.3f},{pout:.3f},{tmp:.1f}", flush=True)
    time.sleep(1)
