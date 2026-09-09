#!/usr/bin/env python3
"""Scan the System Controller's I2C buses for the VCK190 power devices.

Runs on the System Controller (aarch64, BusyBox, no i2c-tools needed): pure
ioctl through /dev/i2c-*. The SC owns the board's PMBus, which is why reads of
the VCCINT regulator from the Versal side return 0xFF.

Prints, per bus, the addresses that acknowledge, and flags the ones we care
about: 0x74 (bus switch), 0x40-0x4f (INA226 rails), 0x16 (IR35215 VCCINT).
Read-only: it probes with a zero-length write, and never writes a register.
"""
import ctypes, fcntl, os, glob
I2C_RDWR, I2C_M_RD = 0x0707, 0x0001
class msg(ctypes.Structure):
    _fields_ = [("addr", ctypes.c_uint16), ("flags", ctypes.c_uint16),
                ("len", ctypes.c_uint16), ("buf", ctypes.POINTER(ctypes.c_uint8))]
class rdwr(ctypes.Structure):
    _fields_ = [("msgs", ctypes.POINTER(msg)), ("nmsgs", ctypes.c_uint32)]

INTEREST = {0x74: "bus switch", 0x16: "IR35215 (VCCINT)", 0x47: "INA226"}
for dev in sorted(glob.glob("/dev/i2c-*")):
    try:
        fd = os.open(dev, os.O_RDWR)
    except OSError as e:
        print(f"{dev}: cannot open ({e.strerror})"); continue
    found = []
    for a in range(0x08, 0x78):
        buf = (ctypes.c_uint8 * 1)()
        m = (msg * 1)(msg(a, I2C_M_RD, 1, buf))
        try:
            fcntl.ioctl(fd, I2C_RDWR, rdwr(m, 1)); found.append(a)
        except OSError:
            pass
    os.close(fd)
    tags = [f"0x{a:02x}" + (f" <- {INTEREST[a]}" if a in INTEREST else
            (" <- INA226?" if 0x40 <= a <= 0x4f else "")) for a in found]
    print(f"{dev}: {len(found)} device(s)" + ("  " + ", ".join(tags) if tags else ""))
