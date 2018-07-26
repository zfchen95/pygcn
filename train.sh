source activate pygcn_dev

for i in {14..14}
do
    for j in {15..18}
    do
        python -W ignore pygcn/train.py --train "2018-06-$j" --test "2018-06-$i" --epochs 1000
    done
done
