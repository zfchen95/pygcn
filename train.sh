source activate pygcn_dev

# train on one day and predict for another
# for i in {17..18}
# do
#     for j in {10..18}
#     do
#         python -W ignore pygcn/train.py --train "2018-06-$j" --test "2018-06-$i" --epochs 1000
#     done
# done

# train on part of the one-day graph and predict for the rest
# for j in {10..18}
#     do
#         python -W ignore pygcn/train.py --train "2018-06-$j" --test "2018-06-$i" --epochs 3000
#     done

# train by hours, predict for the next six
# for i in {0..23}
# do
#     for j in {1..6}
#     do
#         num=$(( i + j ))
#         if [ "$i" -lt 10 ]; then
#             if [ "$num" -lt 10 ]; then
#                 python -W ignore pygcn/train.py --train "2018-06-11_0$i" --test "2018-06-11_0$num" --epochs 1000
#             else
#                 python -W ignore pygcn/train.py --train "2018-06-11_0$i" --test "2018-06-11_$num" --epochs 1000
#             fi
#         else
#             if [ "$num" -lt 24 ]; then
#                 python -W ignore pygcn/train.py --train "2018-06-11_$i" --test "2018-06-11_$num" --epochs 1000
#             fi
#         fi
#     done
# done

# train on part of the one-hour graph and predict for the rest
for i in {0..23}
do
#     for hid in 8 16 32 64 128;
    for dropout in 0.1 0.2 0.3 0.4 0.5;
    do
        if [ "$i" -lt 10 ]; then
#             python -W ignore pygcn/train.py --train "2018-06-11_0$i"  --epochs 1000 --hidden "$hid" --output "result/evaluations_hid_$hid.txt"
            python -W ignore pygcn/train.py --train "2018-06-11_0$i"  --epochs 1000 --dropout "$dropout" --output "result/evaluations_dropout_$dropout.txt"
        else
#             python -W ignore pygcn/train.py --train "2018-06-11_$i"  --epochs 1000 --hidden "$hid" --output "result/evaluations_hid_$hid.txt"
            python -W ignore pygcn/train.py --train "2018-06-11_$i"  --epochs 1000 --dropout "$dropout" --output "result/evaluations_dropout_$dropout.txt"
        fi
    done
done