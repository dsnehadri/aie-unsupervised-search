#!/usr/bin/env python3
"""Read the VCCINT regulator's OWN output telemetry over PMBus (IR35215 at 0x16,
i2c-0 through switch 0x74 channel 0, probed 2026-09-07).

Why: the INA226 at 0x40 senses ONE of the six VCCINT phases and the sampler
multiplies by 6; under light load the controller sheds phases and that phase
reads ~zero or negative. READ_POUT from the controller integrates all phases.

PMBus READ_WORD needs a REPEATED START between the command byte and the read
(a write-then-read with a STOP in between returns 0xFFFF on this part), so the
transfers use the I2C_RDWR combined-transaction ioctl. LINEAR11 decoding.
Run with power_sampler.py stopped: both re-point switch 0x74.
"""
import ctypes, fcntl, os, sys, time
I2C_SLAVE, I2C_RDWR, I2C_M_RD = 0x0703, 0x0707, 0x0001
MUX, CH, ADDR = 0x74, 0, 0x16
class i2c_msg(ctypes.Structure):
    _fields_ = [("addr", ctypes.c_uint16), ("flags", ctypes.c_uint16), ("len", ctypes.c_uint16), ("buf", ctypes.POINTER(ctypes.c_uint8))]
class i2c_rdwr(ctypes.Structure):
    _fields_ = [("msgs", ctypes.POINTER(i2c_msg)), ("nmsgs", ctypes.c_uint32)]
fd = os.open("/dev/i2c-0", os.O_RDWR)
def xfer(addr, wr, nrd):
    wbuf = (ctypes.c_uint8 * len(wr))(*wr); rbuf = (ctypes.c_uint8 * max(nrd, 1))()
    msgs = (i2c_msg * 2)(i2c_msg(addr, 0, len(wr), wbuf), i2c_msg(addr, I2C_M_RD, nrd, rbuf))
    pkt = i2c_rdwr(msgs, 2 if nrd else 1)
    fcntl.ioctl(fd, I2C_RDWR, pkt); return bytes(rbuf[:nrd])
def wr_byte(addr, cmd, val): xfer(addr, [cmd, val], 0)
def rd_word(addr, cmd): b = xfer(addr, [cmd], 2); return b[0] | (b[1] << 8)
def rd_byte(addr, cmd): return xfer(addr, [cmd], 1)[0]
def lin11(w):
    m = w & 0x7FF; e = (w >> 11) & 0x1F
    if m >= 0x400: m -= 0x800
    if e >= 0x10: e -= 0x20
    return m * 2.0 ** e
xfer(MUX, [1 << CH], 0)                              # point the switch at the regulator channel
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
print("t,page,vout_V,iout_A,pout_W,temp_C")
for _ in range(n):
    for p in (0, 1):                                 # IR35215 loops: page 0 and page 1 (VCCINT / VCC_SOC per UG1366)
        wr_byte(ADDR, 0x00, p)
        vm = rd_byte(ADDR, 0x20); e = vm & 0x1F; e = e - 32 if e >= 16 else e
        vout = rd_word(ADDR, 0x8B) * 2.0 ** e
        iout = lin11(rd_word(ADDR, 0x8C)); pout = lin11(rd_word(ADDR, 0x96)); tmp = lin11(rd_word(ADDR, 0x8D))
        print(f"{time.time():.2f},{p},{vout:.4f},{iout:.3f},{pout:.3f},{tmp:.1f}", flush=True)
    time.sleep(1)
