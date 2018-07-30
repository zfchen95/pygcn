source activate pygcn_dev

# for i in {17..18}
# do
#     for j in {10..18}
#     do
#         python -W ignore pygcn/train.py --train "2018-06-$j" --test "2018-06-$i" --epochs 1000
#     done
# done

# for j in {10..18}
#     do
#         python -W ignore pygcn/train.py --train "2018-06-$j" --test "2018-06-$i" --epochs 3000
#     done

for i in {0..23}
do
    for j in {1..6}
    do
#         python -W ignore pygcn/train.py --train "2018-06-$j" --test "2018-06-$i" --epochs 1000
        num=$(( i + j ))
        if [ "$i" -lt 10 ]; then
            if [ "$num" -lt 10 ]; then
                python -W ignore pygcn/train.py --train "2018-06-11_0$i" --test "2018-06-11_0$num" --epochs 1000
            else
                python -W ignore pygcn/train.py --train "2018-06-11_0$i" --test "2018-06-11_$num" --epochs 1000
            fi
        else
            if [ "$num" -lt 24 ]; then
                python -W ignore pygcn/train.py --train "2018-06-11_$i" --test "2018-06-11_$num" --epochs 1000
            fi
        fi
    done
done