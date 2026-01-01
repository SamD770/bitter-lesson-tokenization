from transformers import AutoTokenizer

my_model = ...

my_string = "Hello, world!"

split_index = 7

split_1 = my_string[:split_index]
split_2 = my_string[split_index:]

out_full = my_model(my_string)

out_inter = my_model(split_1)
out_split = my_model(split_2, *out_inter)

print(split_1)
print(split_2)