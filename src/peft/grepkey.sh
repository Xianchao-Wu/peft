
if [ $# -lt 1 ]
then
	echo "Usage: $0 <key>"
	exit 1
fi

key=$1

grep -s $key *.py
grep -s $key */*.py
grep -s $key */*/*.py
grep -s $key */*/*/*.py
grep -s $key */*/*/*/*.py
grep -s $key */*/*/*/*/*.py

grep -s $key *.sh
grep -s $key */*.sh
grep -s $key */*/*.sh
grep -s $key */*/*/*.sh
grep -s $key */*/*/*/*.sh
grep -s $key */*/*/*/*/*.sh
