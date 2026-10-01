import csv
import io

import streamlit as st


SUBJECTS = ("Maths", "Science", "English", "Computer", "Social")
COLUMNS = ("Roll No.", "Name", *SUBJECTS, "Total", "Average", "Grade")


def grade_for(average):
	if average >= 90:
		return "A+"
	if average >= 80:
		return "A"
	if average >= 70:
		return "B"
	if average >= 60:
		return "C"
	if average >= 50:
		return "D"
	return "F"


def make_csv(students):
	buffer = io.StringIO()
	writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
	writer.writeheader()
	writer.writerows(students)
	return buffer.getvalue().encode("utf-8-sig")


st.set_page_config(page_title="Student Grade Manager", page_icon="SG", layout="wide")
st.title("Student Grade Manager")

if "students" not in st.session_state:
	st.session_state.students = []

students = st.session_state.students
averages = [student["Average"] for student in students]
metric_columns = st.columns(3)
metric_columns[0].metric("Students", len(students))
metric_columns[1].metric(
	"Class average", f"{sum(averages) / len(averages):.1f}" if averages else "--"
)
metric_columns[2].metric(
	"Passing", sum(student["Grade"] != "F" for student in students)
)

add_tab, records_tab = st.tabs(("Add student", "Student records"))

with add_tab:
	with st.form("add_student_form", clear_on_submit=True):
		identity_columns = st.columns(2)
		name = identity_columns[0].text_input("Student name")
		roll_number = identity_columns[1].text_input("Roll number")

		mark_columns = st.columns(len(SUBJECTS))
		marks = {
			subject: mark_columns[index].number_input(
				subject,
				min_value=0,
				max_value=100,
				value=0,
				step=1,
			)
			for index, subject in enumerate(SUBJECTS)
		}

		submitted = st.form_submit_button("Add student", type="primary")

	if submitted:
		clean_name = name.strip()
		clean_roll_number = roll_number.strip()
		existing_roll_numbers = {
			student["Roll No."].casefold() for student in students
		}

		if not clean_name or not clean_roll_number:
			st.error("Enter both a student name and roll number.")
		elif clean_roll_number.casefold() in existing_roll_numbers:
			st.error("That roll number is already in the records.")
		else:
			total = sum(marks.values())
			average = total / len(SUBJECTS)
			student = {
				"Roll No.": clean_roll_number,
				"Name": clean_name,
				**marks,
				"Total": total,
				"Average": round(average, 2),
				"Grade": grade_for(average),
			}
			st.session_state.students.append(student)
			st.success(f"Added {clean_name} with grade {student['Grade']}.")

with records_tab:
	if not students:
		st.info("No student records yet.")
	else:
		search = st.text_input("Search by name or roll number")
		query = search.casefold().strip()
		visible_students = [
			student
			for student in students
			if query in student["Name"].casefold()
			or query in student["Roll No."].casefold()
		]

		st.dataframe(
			[{column: student[column] for column in COLUMNS} for student in visible_students],
			hide_index=True,
			use_container_width=True,
		)
		st.download_button(
			"Download CSV",
			data=make_csv(students),
			file_name="student_grades.csv",
			mime="text/csv",
		)

		selected_roll_number = st.selectbox(
			"Remove a student",
			options=[student["Roll No."] for student in students],
		)
		if st.button("Remove selected student"):
			st.session_state.students = [
				student
				for student in students
				if student["Roll No."] != selected_roll_number
			]
			st.rerun()
