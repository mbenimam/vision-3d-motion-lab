#!/usr/bin/env bash
# Download the public datasets the app can use, into data/ (git-ignored). About 80 MB.
# The app works without them (every topic has synthetic inputs with exact ground truth);
# with them you also get real pictures and the reference ground truth of each benchmark.
#
#   scripts/fetch_data.sh            # everything
#   scripts/fetch_data.sh stereo     # one group: stereo homography panorama epipolar flow video
#
# Licences: these datasets are published for research use by their authors; please cite them
# (see data/SOURCES.md, written by this script). They are not redistributed in this repository.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
data="$root/data"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$data"

get() {  # get URL DEST: download once, quietly, failing loudly
  local url="$1" dest="$2"
  [[ -s "$dest" ]] && return 0
  mkdir -p "$(dirname "$dest")"
  echo "  $url"
  curl -fsSL --retry 3 -o "$dest.part" "$url" && mv "$dest.part" "$dest"
}

want() { [[ $# -eq 0 || " ${GROUPS_WANTED[*]} " == *" $1 "* ]]; }
GROUPS_WANTED=("$@")
[[ ${#GROUPS_WANTED[@]} -eq 0 ]] && GROUPS_WANTED=(stereo homography panorama epipolar flow video)

MB=https://vision.middlebury.edu

if want stereo; then
  echo "stereo: Middlebury 2001 (Tsukuba) and 2003 (Cones, Teddy), quarter size, with ground truth"
  d="$data/stereo/tsukuba"
  get "$MB/stereo/data/scenes2001/data/tsukuba/scene1.row3.col3.ppm" "$d/left.ppm"
  get "$MB/stereo/data/scenes2001/data/tsukuba/scene1.row3.col4.ppm" "$d/right.ppm"
  get "$MB/stereo/data/scenes2001/data/tsukuba/truedisp.row3.col3.pgm" "$d/truth.pgm"
  echo 16 > "$d/scale.txt"   # stored value = disparity x 16
  for s in cones teddy; do
    d="$data/stereo/$s"
    get "$MB/stereo/data/scenes2003/newdata/$s/im2.png" "$d/left.png"
    get "$MB/stereo/data/scenes2003/newdata/$s/im6.png" "$d/right.png"
    get "$MB/stereo/data/scenes2003/newdata/$s/disp2.png" "$d/truth.png"
    echo 4 > "$d/scale.txt"  # stored value = disparity x 4
  done
fi

if want homography; then
  echo "homography: Oxford affine 'graf' sequence (planar scene, ground-truth homographies)"
  get "https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz" "$tmp/graf.tar.gz"
  mkdir -p "$data/homography/graf" && tar xzf "$tmp/graf.tar.gz" -C "$data/homography/graf"
fi

if want panorama; then
  echo "panorama: OpenCV test images for stitching"
  for i in 1 2 3; do
    get "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/stitching/a$i.png" "$data/panorama/a$i.png"
  done
fi

if want epipolar; then
  echo "epipolar: Oxford VGG 'Merton College I' (three views of a building)"
  get "https://www.robots.ox.ac.uk/~vgg/data/merton.a/images.tar.gz" "$tmp/merton.tar.gz"
  mkdir -p "$data/epipolar/merton" && tar xzf "$tmp/merton.tar.gz" -C "$data/epipolar/merton"
fi

if want flow; then
  echo "flow: Middlebury optical flow 'other' sequences, two frames with ground-truth flow"
  get "$MB/flow/data/comp/zip/other-color-twoframes.zip" "$tmp/frames.zip"
  get "$MB/flow/data/comp/zip/other-gt-flow.zip" "$tmp/gt.zip"
  mkdir -p "$data/flow" && unzip -qo "$tmp/frames.zip" -d "$data/flow" && unzip -qo "$tmp/gt.zip" -d "$data/flow"
fi

if want video; then
  echo "video: test videos of the ENSTA tracking practical (A. Manzanera)"
  get "https://perso.ensta-paris.fr/~manzaner/Cours/IMA/VISION/Test-Videos.zip" "$tmp/videos.zip"
  mkdir -p "$data/video" && unzip -qo "$tmp/videos.zip" -d "$data/video"
fi

cat > "$data/SOURCES.md" <<'EOF'
# Data sources (downloaded by scripts/fetch_data.sh, not redistributed)

- Stereo: D. Scharstein and R. Szeliski, "A taxonomy and evaluation of dense two-frame stereo
  correspondence algorithms", IJCV 2002 (Tsukuba); "High-accuracy stereo depth maps using
  structured light", CVPR 2003 (Cones, Teddy). https://vision.middlebury.edu/stereo/
- Optical flow: S. Baker et al., "A database and evaluation methodology for optical flow",
  IJCV 2011. https://vision.middlebury.edu/flow/
- Homography: K. Mikolajczyk et al., "A comparison of affine region detectors", IJCV 2005
  (graf sequence). https://www.robots.ox.ac.uk/~vgg/research/affine/
- Epipolar: Oxford Visual Geometry Group multi-view data, Merton College I.
  https://www.robots.ox.ac.uk/~vgg/data/mview/
- Panorama: OpenCV test data (opencv_extra, testdata/stitching).
- Videos: A. Manzanera, ENSTA Paris, UE Vision practical work on tracking.
EOF
echo "done: $(du -sh "$data" | cut -f1) in $data"
