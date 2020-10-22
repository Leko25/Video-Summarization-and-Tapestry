#!/bin/bash
working_dir=`pwd`
if [ -d "$working_dir/metadata" ]; then
  rm -rf $working_dir/metadata
fi

mkdir metadata


filenames=`ls ./videos/ | sort -n`
counter=1
for file in $filenames
do
	echo "Processing $file"
	mkdir metadata/video_$counter
	current_dir=metadata/video_$counter
	python3 $working_dir/HelperScripts/SynopsisSummarization/tester.py $working_dir/videos/$file $current_dir/extracted $current_dir $current_dir"_synop"
	cd metadata
	temp_files=video_$counter"_"*
	for temp_file in $temp_files
	do
		mv $temp_file video_$counter
	done
	((counter=counter+1))
	cd ..
done
