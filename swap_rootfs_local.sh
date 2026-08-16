#!/bin/bash
# swap_rootfs_local.sh — run on YOUR OWN machine (with sudo + a card reader) to
# replace the VCK190 SD card's rootfs (partition 2) with the 2025.2 PetaLinux
# rootfs. Leaves partition 1 (FAT boot) UNTOUCHED. Self-contained: only needs
# the rootfs tarball you copy over.
#
# Usage:  sudo ./swap_rootfs_local.sh <rootfs.tar.gz> /dev/sdXN
#         (N = the ext4 ROOT partition, ~13.9 GB — NOT the 976 MB FAT one)
#
# Rollback: the bit-exact whole-card image lives on millerlabml01 at
#   /home/snehadri/board_full_backup/2026-06-11_pre_rootfs_swap/mmcblk0_full.img.gz
#   restore with:  gunzip -c mmcblk0_full.img.gz | sudo dd of=/dev/sdX bs=4M status=progress conv=fsync

set -euo pipefail
die(){ echo "ERROR: $*" >&2; exit 1; }

[ $# -eq 2 ] || die "usage: sudo $0 <rootfs.tar.gz> /dev/sdXN   (sdXN = ext4 root partition, ~13.9 GB)"
TGZ="$1"; PART="$2"
[ "$(id -u)" -eq 0 ] || die "must run as root (sudo)"
[ -f "$TGZ" ] || die "rootfs tarball not found: $TGZ"
[ -b "$PART" ] || die "$PART is not a block device"

# Guard: never touch the disk that holds THIS machine's running root filesystem.
PKNAME=$(lsblk -no PKNAME "$PART" 2>/dev/null | head -1)
[ -n "$PKNAME" ] || die "could not determine parent disk of $PART"
HOST_ROOT_SRC=$(findmnt -no SOURCE / 2>/dev/null || true)
HOST_ROOT_DISK=$(lsblk -no PKNAME "$HOST_ROOT_SRC" 2>/dev/null | head -1 || true)
[ -n "$HOST_ROOT_DISK" ] && [ "$PKNAME" = "$HOST_ROOT_DISK" ] && die "$PART is on THIS machine's system disk (/dev/$HOST_ROOT_DISK) — REFUSING"

# Sanity: board p2 is ~13.9 GB ext4, partition #2.
SZ=$(blockdev --getsize64 "$PART"); SZ_GB=$(( SZ / 1000000000 ))
echo "Target partition : $PART  (parent /dev/$PKNAME)"
echo "Partition size   : ${SZ_GB} GB"
echo "Parent disk partition table:"
sfdisk -l "/dev/$PKNAME" 2>/dev/null | sed 's/^/    /' || lsblk "/dev/$PKNAME"
[ "$SZ_GB" -ge 11 ] && [ "$SZ_GB" -le 15 ] || die "partition is ${SZ_GB} GB; board p2 is ~13.9 GB. Wrong partition? REFUSING."
echo "$PART" | grep -qE '[p]?2$' || echo "WARNING: '$PART' doesn't look like partition #2 — make sure this is ROOT (p2), NOT the 976MB FAT boot (p1)."

# Confirm the FAT boot partition (p1) still looks right (read-only peek, optional)
P1="${PART%2}1"; [ "$PART" = "${PART%p2}p2" ] && P1="${PART%p2}p1"
echo "Sibling FAT boot partition expected at: $P1 (will NOT be modified)"

for mp in $(findmnt -nro TARGET "$PART" 2>/dev/null || true); do echo "unmounting $mp"; umount "$mp" || die "cannot unmount $mp"; done

echo
echo "About to: mkfs.ext4 $PART  +  extract $(basename "$TGZ")."
echo "Partition 1 (FAT boot) NOT touched."
read -r -p "Type EXACTLY 'SWAP $PART' to proceed: " CONF
[ "$CONF" = "SWAP $PART" ] || die "confirmation mismatch — aborting (nothing changed)"

MNT=$(mktemp -d)
echo "[1/4] mkfs.ext4 on $PART ..."; mkfs.ext4 -F -L root "$PART"
echo "[2/4] mount ..."; mount "$PART" "$MNT"
echo "[3/4] extracting 2025.2 rootfs (1-2 min) ..."; tar --numeric-owner -xpzf "$TGZ" -C "$MNT"
echo "[4/4] sync + unmount ..."; sync; umount "$MNT"; sync; rmdir "$MNT"

echo
echo "DONE. 2025.2 rootfs written to $PART (FAT boot p1 untouched)."
echo "Reinsert the card in the VCK190 and power it on."
echo "Your /home/root + /root work files were NOT restored here (they're backed up on"
echo "millerlabml01); restore them after boot if you want — see note from Claude."
