# Execute the command that generates the list of PIDs and loop through each PID
top -n 1 -b -u $USER | grep "python" | grep -v grep | awk '{print $1}' | while read -r pid; do
    ls /proc/$pid/task | xargs renice 6
done