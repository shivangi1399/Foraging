#!/bin/bash

# Find directories modified between 10 and 180 minutes ago and larger than 1000 MB
mapfile -d $'\0' dirs < <(find /mnt/as/projects/OWzeronoise/ -maxdepth 5 -mindepth 5 -cmin +10 -cmin -180 -print0 | while IFS= read -r -d '' file; do if [ $(du -sh -BM "$file" | cut -f1 | tr -d 'M') -ge 1000 ]; then echo -ne "$file\0"; fi; done)

# If no such directory is found, exit the script
if [ ${#dirs[@]} -eq 0 ]; then
    echo "Nothing to do... NEXT!"
    exit
fi

# For each found directory, check if it contains an empty "sync_messages.txt" file
for dir in "${dirs[@]}"; do
    if test $(find "$dir" -cmin -60 -empty -name "sync_messages.txt" | wc -c) -ne 0; then
        echo "The file below is empty. Please correct before it goes to tape archive."
        echo "For a tutorial on how to do this, see http://intranet.esi.local/confluence/display/HSV/What+to+do+when+sync_messages.txt+is+empty"
        find "$dir" -cmin -60 -empty -name "sync_messages.txt"
        echo
    fi
done | /usr/bin/mail -E -s "empty recording file" muad.abd-el-hay@esi-frankfurt.de

