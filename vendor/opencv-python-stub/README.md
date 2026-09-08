# opencv-python stub

`ultralytics` hard-requires `opencv-python>=4.6.0` even though this project only
wants `opencv-python-headless` (no GUI/GL bindings, no need for `libGL.so.1`).

If both packages are installed for real, they write to the same `cv2/`
directory in site-packages and whichever installs last wins — unreliable, and
on Streamlit Community Cloud the real `opencv-python` needs system libraries
(`libGL.so.1`) that `packages.txt` can't currently provide (the base image's
apt sources are broken).

This directory is an empty package that identifies itself as `opencv-python`
to pip so it satisfies ultralytics' dependency check without installing any
code or native binaries. The actual `cv2` module always comes from
`opencv-python-headless`, listed separately in `requirements.txt`.
