scons

timeout=0
timeoutlimit=1
time=0

for i in {1..1}
do
    time=$((time + 1))
    if ./bin/gosdt experiments/datasets/test.csv experiments/configurations/debug.json > ./output.txt && grep -q 'timeout' './output.txt';
    then
        timeout=$((timeout + 1))
        cp ./output.txt ./timeout.txt
        if [ $timeout -ge $timeoutlimit ]
        then
            break
        fi
    fi
done

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

msg=""
if [ $timeout -gt 0 ]
then
    msg="Time out $timeout times out of $time times!"
    /usr/bin/osascript -e "display notification \"$msg\" with title \"Test Failure\" sound name \"Sosumi\""
    cp ./timeout.txt ./output.txt
    rm ./timeout.txt
    msg="$RED$msg$NC"
else
    msg="Test ran successfully $time times!"
    /usr/bin/osascript -e "display notification \"$msg\" with title \"Test Success\" sound name \"Blow\""
    msg="$GREEN$msg$NC"
fi

echo -e $msg

# scons && ./bin/gosdt experiments/datasets/test.csv experiments/configurations/debug.json