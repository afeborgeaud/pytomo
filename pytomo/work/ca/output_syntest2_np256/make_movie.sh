#!/bin/sh
ffmpeg -r 4 -i voronoi_%05d.png -c:v libx264 -pix_fmt yuv420p -y -an -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white" result.mp4
