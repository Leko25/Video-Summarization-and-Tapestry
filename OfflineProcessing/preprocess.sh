#!/bin/bash

javac ./HelperScripts/ConvertRGBToPNG/ConvertRGBToPNG.java

counter=1

working_dir=`pwd`
filenames=`ls ./RawRGBFiles/`
for file in $filenames
do
	file_name="Output$counter/";
	echo "$file";
	cd ./RawRGBFiles/$file
	images_dir=`pwd`
	mkdir RGBimages
	cp *.rgb RGBimages
	mkdir PNGimages
	cd $working_dir/HelperScripts/ConvertRGBToPNG
	java ConvertRGBToPNG $images_dir/RGBimages $images_dir/PNGimages
	cd $images_dir
	sound_file=`ls | grep wav`
	rm -rf RGBimages 
	cd $working_dir
	output_video="video_processed$counter.mp4"
	python $working_dir/HelperScripts/ConvertPNGToMP4/driver.py $images_dir/PNGimages $images_dir/$sound_file $working_dir/$output_video
	rm -rf $images_dir/PNGimages
	((counter=counter+1));
done
cd $working_dir

if [ -d "$working_dir/videos" ]; then
  rm -rf $working_dir/videos
fi

mkdir videos

mv *.mp4 ./videos
cp ./videos/* ./resources/video/
