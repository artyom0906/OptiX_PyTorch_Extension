## GPU Performance & Configuration Report
Date: Mon Mar  3 07:53:10 PM EET 2025

### PCIe Configuration
```
            PCIe Generation
                Max                       : 5
                Current                   : 5
                Device Current            : 5
                Device Max                : 5
                Host Max                  : 5
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 500423 KB/s
        Rx Throughput                     : 798920 KB/s
        Atomic Caps Outbound              : N/A
        Atomic Caps Inbound               : N/A
    Fan Speed                             : 30 %
    Performance State                     : P1
    Clocks Event Reasons
--
            PCIe Generation
                Max                       : 3
                Current                   : 3
                Device Current            : 3
                Device Max                : 5
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 1x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 315885 KB/s
        Rx Throughput                     : 2685 KB/s
        Atomic Caps Outbound              : N/A
        Atomic Caps Inbound               : N/A
    Fan Speed                             : 0 %
    Performance State                     : P1
    Clocks Event Reasons
```

### System Information
```
00:02.0 Display controller: Intel Corporation AlderLake-S GT1 (rev 0c)
01:00.0 VGA compatible controller: NVIDIA Corporation GB202 [GeForce RTX 5090] (rev a1)
05:00.0 VGA compatible controller: NVIDIA Corporation GB202 [GeForce RTX 5090] (rev a1)
```

### GPU Memory Information
```
index, name, memory.total [MiB], memory.used [MiB], memory.free [MiB]
0, NVIDIA GeForce RTX 5090, 32607 MiB, 3816 MiB, 28283 MiB
1, NVIDIA GeForce RTX 5090, 32607 MiB, 2112 MiB, 30010 MiB
```

### Motherboard PCIe Slots
```
-[0000:00]-+-00.0
           +-01.0-[01]--+-00.0
           |            \-00.1
           +-02.0
           +-06.0-[02]----00.0
           +-08.0
           +-0a.0
           +-14.0
           +-14.2
           +-14.3
           +-16.0
           +-17.0
           +-1a.0-[03]----00.0
           +-1c.0-[04]--
           +-1c.1-[05]--+-00.0
           |            \-00.1
           +-1c.3-[06]----00.0
           +-1f.0
           +-1f.3
           +-1f.4
           \-1f.5
```

### Performance Analysis

The significant render time difference (1.4-2ms for GPU 0 vs. 7-7.5ms for GPU 1) is likely
explained by the PCIe configuration differences:

**GPU 0 (faster):**
- PCIe Generation: 5
- Link Width: 16x
- Bus ID: 01:00.0

**GPU 1 (slower):**
- PCIe Generation: 3 (instead of 5)
- Link Width: 1x (instead of 16x)
- Bus ID: 05:00.0

The second GPU is running at PCIe Gen 3 x1, which is only ~3.1% of the maximum bandwidth
available to GPU 0 (PCIe Gen 5 x16). This explains the approximately 3.5-5x performance
difference in rendering time.

### Recommendation

1. Verify your motherboard BIOS settings - check if there are options to configure PCIe lanes
   or bifurcation settings for the second GPU slot.
2. Check if GPU 1 is installed in a slot that's intended to run at x16 width.
3. Ensure that no PCIe lanes are being shared with other devices (like NVMe drives).
4. If possible, try moving GPU 1 to a different PCIe slot with more dedicated lanes.
