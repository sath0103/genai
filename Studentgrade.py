students = []

while True:
    print("\n===== Student Management System =====")
    print("1. Add Student")
    print("2. Display Students")
    print("3. Search Student")
    print("4. Exit")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        name = input("Enter student name: ")
        age = int(input("Enter age: "))
        mark = float(input("Enter mark: "))

        student = {
            "name": name,
            "age": age,
            "mark": mark
        }

        students.append(student)
        print("Student added successfully!")

    elif choice == 2:
        if len(students) == 0:
            print("No students found.")
        else:
            print("\n--- Student Details ---")
            for student in students:
                print("Name:", student["name"])
                print("Age:", student["age"])
                print("Mark:", student["mark"])
                print("----------------------")

    elif choice == 3:
        search_name = input("Enter student name to search: ")

        found = False

        for student in students:
            if student["name"].lower() == search_name.lower():
                print("\nStudent Found!")
                print("Name:", student["name"])
                print("Age:", student["age"])
                print("Mark:", student["mark"])
                found = True
                break

        if not found:
            print("Student not found.")

    elif choice == 4:
        print("Program ended.")
        break

    else:
        print("Invalid choice!")