#!/bin/sh
# Fetch the test clips and convert them to the raw float32 the runner reads.
#
# The clips are the RNNoise demo samples. They are downloaded from their
# original location rather than redistributed here, so the terms are whatever
# jmvalin.ca states; this repository claims nothing over them.
#
# Requires ffmpeg. Output is 48 kHz mono float32-LE, headerless.
set -e
cd "$(dirname "$0")"

BASE=https://jmvalin.ca/demo/rnnoise/samples

for name in babble_5dB car_5dB street_5dB; do
    if [ ! -f "$name.opus" ]; then
        echo "fetching $name.opus"
        curl -fsSL -o "$name.opus" "$BASE/$name.opus"
    fi
    echo "decoding $name.f32"
    ffmpeg -loglevel error -y -i "$name.opus" \
        -f f32le -acodec pcm_f32le -ac 1 -ar 48000 "$name.f32"
done

echo "done. try:"
echo "  ./build/fe_run_file weights/fe.q8 testaudio/street_5dB.f32 out.wav 5569 i8mm"
