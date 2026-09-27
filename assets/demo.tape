# Renders assets/demo.gif. Re-record with: vhs assets/demo.tape (from the repo root)
Output assets/demo.gif

Set Shell bash
Set FontSize 14
Set Width 1180
Set Height 300
Set Padding 16
Set Theme "Catppuccin Mocha"
Set Framerate 10
Set PlaybackSpeed 2

# Start off-camera so the GIF opens on the UI, not on typing.
Hide
Type "clear && gpmtop --demo"
Enter
Sleep 3s
Show

# One full demo cycle: compute, memory, stop-start, launch-bound, FP32.
Sleep 97s
