import os, json
import shutil



def unbalanced_preprocess():
	# info_path=r'D:\locs\data\images\face_age_gender\Eastern_people\tarball\AFAD-Full\AFAD-Full.txt'
	# file =  open(info_path, 'r')

	i=0
	total_info = []
	male_count = 0
	female_count = 0
	total_count={}
	dir_='dataset/train_image_unbalanced/'
	os.makedirs(dir_, exist_ok=True)
	for (_dir, _, files) in os.walk(r'dataset/tarball-with_mask_unbalanced'):
		for file in files:
			i += 1
			if not file.endswith('.jpg'):
				continue
			file_path = os.path.join(_dir, file)
			print(file_path)
			age, gender, filename = file_path.split('/')[-3:]
			age = (int(age)//10) * 10
			gender = "male" if gender=="111" else "female"
			new_path = os.path.join(dir_, filename)
			dict_ = {"image": new_path.strip(), "age": age, "gender": gender}

			if gender == "male":
				if female_count >= male_count:
					male_count+=1
					total_info.append(dict_)
					if age in total_count.keys():
						total_count[age]+=1
					else:
						total_count[age]=1
			else:
				female_count+=1
				total_info.append(dict_)
				if age in total_count.keys():
					total_count[age] += 1
				else:
					total_count[age] = 1
			shutil.copy(file_path.strip(), dir_)
	print(total_count)
	print(f'male: {male_count}')
	print(f'female: {female_count}')
	with open(os.path.join(dir_,'metadata.json'), 'w') as info_file:
		info_file.write(str(total_info).replace('\'','\"'))
	print(f'Finished..')


def unbalanced_aihub_preprocess():
	i = 0
	total_info = []
	male_count = 0
	female_count = 0
	total_count = {}
	dir_ = 'dataset/aihub_unbalanced'
	os.makedirs(dir_, exist_ok=True)
	for (dir__, _, files) in os.walk(r'dataset/extracted_aihub'):
		for file in files:
			i += 1
			if not file.endswith('.jpg'):
				continue

			file_path = os.path.join(dir__, file)
			age, gender = file_path.replace('.jpg','').split('_')[-2:]
			age = int(age)
			new_path = os.path.join(dir_, file)
			dict_ = {"image": new_path.strip(), "age": age, "gender": gender}

			if gender == "male":
				if female_count >= male_count:
					male_count += 1
					total_info.append(dict_)
					if age in total_count.keys():
						total_count[age] += 1
					else:
						total_count[age] = 1
			else:
				female_count += 1
				total_info.append(dict_)
				if age in total_count.keys():
					total_count[age] += 1
				else:
					total_count[age] = 1
			shutil.copy(file_path.strip(), dir_)
	print(total_count)
	with open(os.path.join(dir_, 'metadata.json'), 'w') as info_file:
		info_file.write(str(total_info).replace('\'', '\"'))
	print(f'Finished..')





def balanced_preprocess():
	max_=3000
	f_count=0
	m_count=0
	total_info = []
	male_count = {}
	female_count = {}
	dir_='dataset/train_image/'
	os.makedirs(dir_, exist_ok=True)
	i=0
	for (dir,_,files) in os.walk(r'dataset/tarball-with_mask_unbalanced'):
		for file in files:
			i += 1
			print(i)
			if not file.endswith('.jpg'):
				continue
			file_path = os.path.join(dir, file)
			#print(file_path)
			age, gender, filename = file_path.split('/')[-3:]


			age = int(age)
			gender = "male" if gender=="111" else "female"
			dict_ = {"image": file_path.strip(), "age": age, "gender": gender}

			if gender == "male":
				if gender in male_count.keys():
					if age in male_count[gender].keys():
						if male_count[gender][age] < max_:
							male_count[gender][age] +=1
							total_info.append(dict_)
							m_count += 1
							shutil.copy(file_path.strip(), 'dataset/train_image/')


						else:
							continue
					else:
						male_count[gender][age]=1
						total_info.append(dict_)
						m_count += 1

						shutil.copy(file_path.strip(), 'dataset/train_image/')


				else:
					male_count[gender] = {}
					male_count[gender][age] =1
					total_info.append(dict_)
					m_count += 1

					shutil.copy(file_path.strip(), 'dataset/train_image/')


			else:

				if gender in female_count.keys():
					if age in female_count[gender].keys():
						if female_count[gender][age] < max_:
							female_count[gender][age] += 1
							total_info.append(dict_)
							f_count += 1

							shutil.copy(file_path.strip(), 'dataset/train_image/')

						else:
							continue
					else:
						female_count[gender][age] = 1
						total_info.append(dict_)
						f_count += 1

						shutil.copy(file_path.strip(), 'dataset/train_image/')

				else:
					female_count[gender] = {}
					female_count[gender][age] = 1
					total_info.append(dict_)
					f_count += 1

					shutil.copy(file_path.strip(), 'dataset/train_image/')

	print(f'f_count: {f_count}')
	print(f'm_count: {m_count}')
	with open(os.path.join(dir_,'metadata.json'), 'w') as info_file:
		info_file.write(str(total_info).replace('\'','\"'))
	print(f'Finished..')
	#
	# img_path=r'D:\locs\data\images\face_age_gender\Eastern_people\tarball\AFAD-Full'
	#
	# for folder in os.listdir(img_path):
	# 	if '.' in folder:
	# 		continue
	# 	for gender_folder in os.listdir(img_path + '\\' +folder):
	# 		for image in os.listdir(img_path + '\\' +folder +'\\' +gender_folder):
	# 			if image =='Thumbs.db':
	# 				continue
	# 			print(image)

if __name__=='__main__':
	unbalanced_aihub_preprocess()
