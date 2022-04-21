for file in $(ls pycsou*.py); do
echo "$file : $(python3 $file $1 $2)" >> "output${1}_${2}.txt";
done

for file in $(ls pmb*.py); do
echo "$file : $(python3 $file $1 $2)" >> "output${1}_${2}.txt";
done
