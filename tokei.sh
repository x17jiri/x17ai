#!/bin/sh
echo tokei -C -e tests -e main.rs src
tokei -C -e tests -e main.rs src
echo tokei -C -e tests src/new
tokei -C -e tests src/new
