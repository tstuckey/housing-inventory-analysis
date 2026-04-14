#!/bin/bash
notebook_dir="../notebooks/"
param_file_prefix='convenience_routines'
logging_enabled=1

# We're expecting exactly one parameter, the name of the database,
# and that it ends in .db
if [[ "${#}" -ne 1 ]]; then
  echo "Usage: update_db_version.sh ./_data/db_name.db"
  exit 0 
elif [[ ${1} != *.db ]]; then
  echo "Enter a valid .db name"
  exit 1
fi

# Get the md5 checksum of the database file; it's in the first field of output
my_sum=$(md5sum "${1}" | cut -d ' ' -f1)

# Get just the name of the database by trimming off the path
db_name=$(basename "${1}")
echo "${db_name}"

if [[ ${logging_enabled} -eq 1 ]]; then 
  printf "Database name is %s and hash is %s \\n \\n" "${1}" "${my_sum}"
fi

# Get the files with db information and cast the result as an array
# shellcheck disable=SC2207
db_param_files=( $(ls "${notebook_dir}""${param_file_prefix}"*) )

if [[ ${logging_enabled} -eq 1 ]]; then
  printf "db_param_files:\n"
  printf "%s | " "${db_param_files[@]}"
  printf "\n\n"
fi

# Iterate through each file that starts with the param_file_prefix
for my_file in "${db_param_files[@]}"
do
  if [[ ${logging_enabled} -eq 1 ]]; then 
    printf "Processing %s\t" "${my_file}"
    printf "\n\n"
  fi 
  
  if [[ ${my_file} =~ py$ ]]; then
    # Handle the Python helper file
    # Update the database name
    sed "s/db_name = .*/db_name = '${db_name}'/g" "${my_file}" > "${my_file}_tmp1"
    # Update the checksum
    sed "s/md5_val = .*/md5_val = '${my_sum}'/g" "${my_file}_tmp1" > "${my_file}_tmp2"
    mv "${my_file}_tmp2" "${my_file}"
    rm "${my_file}_tmp1"
  fi

  if [[ ${my_file} =~ R$ ]]; then
    # Handle the R helper file
    # Update the database name
    sed "s/db_name <- .*/db_name <- '${db_name}'/g" "${my_file}" > "${my_file}_tmp1"
    # Update the checksum
    sed "s/md5_val <- .*/md5_val <- '${my_sum}'/g" "${my_file}_tmp1" > "${my_file}_tmp2"
    mv "${my_file}_tmp2" "${my_file}"
    rm "${my_file}_tmp1"
  fi
done
