# ece544NailA
You have to download classes from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap
https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified
https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/binary

5 classes we are using: eye, foot, finger, hand, leg.

After downloading these files, change the corresponding names to 'eye.npy', 'foot.npy', 'finger.npy', 'hand.npy', 'leg.npy'
'eye.ndjson', 'foot.ndjson', 'finger.ndjson', 'hand.ndjson', 'leg.ndjson'
'eye.bin', 'foot.bin', 'finger.bin', 'hand.bin', 'leg.bin'.

data needed from the front end:
strokes (x, y, p1, p2, p3)
    x, y is the coordinate of a point after converted to a 256x256 canvas. p1: pen is still drawing; p2: pen leaves the drawing panel; p3: drawing finished (time is up)




