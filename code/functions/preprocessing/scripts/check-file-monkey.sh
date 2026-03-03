#!/bin/bash

if test $(find /mnt/as/projects/MWzeronoise/ -cmin -60 -empty -name "sync_messages.txt" | wc -c) -eq 0
    then
     echo "All files fine!"
     exit
else 
{
    echo "The file below is empty. Please correct before it goes to tape archive."
    echo "For a tutorial on how to do this, see http://intranet.esi.local/confluence/display/HSV/What+to+do+when+sync_messages.txt+is+empty"
    find /mnt/as/projects/MWzeronoise/ -cmin -60 -empty -name "sync_messages.txt" 
} | /usr/bin/mail -e -s "empty recording file" muad.abd-el-hay@esi-frankfurt.de
fi

