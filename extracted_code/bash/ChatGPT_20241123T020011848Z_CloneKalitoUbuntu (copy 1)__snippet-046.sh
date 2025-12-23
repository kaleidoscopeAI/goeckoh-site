# Create final directory structure
mkdir -p core functionality scripts/{building-blocks,modules,main_scripts,test,docs,visualization,data}

# Move specific files into their appropriate directories
mv building-block/*.py "core functionality scripts/building-blocks/"
mv modules/*.py "core functionality scripts/modules/"
mv main_scripts/*.py "core functionality scripts/main_scripts/"
mv test/*.py "core functionality scripts/test/"
mv data/* "core functionality scripts/data/"
mv visualization/*.py "core functionality scripts/visualization/"

