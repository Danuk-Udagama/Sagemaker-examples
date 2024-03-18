#!/bin/sh

image=$1

cd ..

docker image rm ${image}
docker build -t ${image} .

cd local_test

mkdir -p test_dir/model
mkdir -p test_dir/output

rm test_dir/model/*
rm test_dir/output/*

docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} train
docker run -v $(pwd)/test_dir:/opt/ml -p 8080:8080 --rm ${image} serve

