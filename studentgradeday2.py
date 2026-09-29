name = input("Enter student name: ")

maths = float(input("Enter Maths mark: "))
science = float(input("Enter Science mark: "))
english = float(input("Enter English mark: "))
computer = float(input("Enter Computer mark: "))
social = float(input("Enter Social mark: "))

total = maths + science + english + computer + social
average = total / 5

if average >= 90:
    grade = "A"
elif average >= 80:
    grade = "B"
elif average >= 70:
    grade = "C"
elif average >= 60:
    grade = "D"
else:
    grade = "F"

print("\n===== Student Result =====")
print("Name:", name)
print("Total:", total)
print("Average:", average)
print("Grade:", grade)