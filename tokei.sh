#!/bin/sh
echo tokei -C -e tests -e main.rs src
tokei -C -e tests -e main.rs src
