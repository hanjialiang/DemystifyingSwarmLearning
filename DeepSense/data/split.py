import os
import random
import json
import shutil
# import hashlib

from tools.mathematics import combine 

_valid_suffix = ['bmp','jpg','jpeg','png','tif','gif','pgm','ppm']
_suffix       = ['a', 'b', 'c', 'd', 'e', 'f']


def process_uncertain(data_path):
	"""
	prepross uncertain directory.
	"""
	img_directory = os.listdir(data_path)
	for dir_name in img_directory:
		if dir_name.lower() == 'ok':
			ok_dir_name = dir_name
			break

	# Process Uncertain Directory
	for class_label in img_directory:
		if class_label.lower() in ["uncertain", "uncertains"]:
			print("uncertain directory checked !")
			uncer_path = os.path.join(data_path, class_label)
			uncer_sub_dir = os.listdir(uncer_path)
			for dir_name in uncer_sub_dir:
				if '-' in dir_name or '_' in dir_name or dir_name in img_directory:
					# move images to class dir
					if dir_name in img_directory:
						taregt_dir = dir_name
					else:
						taregt_dir = dir_name.split('-')[-1] if '-' in dir_name else dir_name.split('_')[-1]
					for img in os.listdir(os.path.join(uncer_path, dir_name)):
						if img.split('.')[-1] in _valid_suffix:
							shutil.move(os.path.join(uncer_path, dir_name, img), os.path.join(data_path, taregt_dir))
				else:
					# move to ok for other situations
					for img in os.listdir(os.path.join(uncer_path, dir_name)):
						if img.split('.')[-1] in _valid_suffix:
							shutil.move(os.path.join(uncer_path, dir_name, img), os.path.join(data_path, ok_dir_name))

			shutil.rmtree(os.path.join(data_path, class_label))
	return 1


def simple_imgfolder(config, label_index, training = True):
	"""
	args:
		<data_root>/
					class1/*.jpg
					class2/*.jpg
					class3/*.jpg
					...
	return:
		[(pos_image, label),...], [(neg_image, label),...](for train), [(pos_image, label),...], [(neg_image, label),...](for validation) in training mode.
	"""
	root = config.data_path
	if not os.path.exists(root):
		print("Root path does not exist, please check your data folder !")
		exit()

	if training:
		train_pos = []
		train_neg = []
		val_pos   = []
		val_neg   = []
		val_rate = config.validation_rate
		assert 1>val_rate and val_rate>0,"validation_rate must be in (0, 1) !"
		if config.del_uncertain:
			process_uncertain(root)
		# Split pos samples and neg samples w.r.t 
		for label in os.listdir(root):
			if label in label_index:
				label_id   = label_index[label]
				label_imgs = []
				for img in os.listdir(os.path.join(root, label)):
					if img.split('.')[-1] in _valid_suffix:
						label_imgs.append((os.path.join(root, label, img), label_id))
				
				# sample image
				for_val   = random.sample(label_imgs, int(val_rate * len(label_imgs)))
				for_train = list(set(label_imgs) - set(for_val))
				if label.lower() == 'ok':
					train_pos.extend(for_train)
					val_pos.extend(for_val)
				else:
					train_neg.extend(for_train)
					val_neg.extend(for_val)
		return train_pos, train_neg, val_pos, val_neg
	else:
		# Construct simple folder
		all_image  = []
		top_file   = os.listdir(root)
		_is_folder = True if top_file[0] in label_index or '_' in top_file[0] else False

		if _is_folder:
			for dir_name in top_file:
				if '_' in dir_name:
					label = dir_name.split('_')[-1]
				else:
					label = dir_name

				if label.lower() != 'uncertain' and label in label_index:
					class_img = []
					label_id = label_index[label]
					for img in os.listdir(os.path.join(root, dir_name)):
						if img.split('.')[-1] in _valid_suffix:
							class_img.append((os.path.join(root, dir_name, img), label_id))

					if hasattr(config, 'k_fold'):
						# split data by class
						if 'k_fold_image' not in dir():
							k_fold_image = [[] for _ in range(config.k_fold)]
						random.shuffle(class_img)
						split_n = max(1, len(class_img)//5)
						for i in range(config.k_fold):
							k_fold_image[i].extend(class_img[split_n*i:split_n*(i+1)])
					else:
						all_image.extend(class_img)
			if hasattr(config, 'k_fold'):
				return k_fold_image
			else:
				return all_image
		else:
			# Unknown image label.
			for img in top_file:
				if img.split('.')[-1] in _valid_suffix:
					all_image.append((os.path.join(root, img), -1))
			return all_image


def trainval_folder(config, label_index, training = True):
	"""
	Compatible with manual training style
	args:
		<data_root>/
			train:
				class1/*.jpg
				class2/*.jpg
				...
			test:
				class1/*.jpg
				class2/*.jpg
				...
	"""
	if training:
		_data_root = config.data_path
		config.data_path = os.path.join(_data_root, 'train')
		_train_pos, _train_neg, _val_pos, _val_neg = simple_imgfolder(config, label_index, training = True)
		train_pos = _train_pos + _val_pos
		train_neg = _train_neg + _val_neg
		
		config.data_path = os.path.join(_data_root, 'test')
		_train_pos, _train_neg, _val_pos, _val_neg = simple_imgfolder(config, label_index, training = True)
		val_pos   = _train_pos + _val_pos
		val_neg   = _train_neg + _val_neg

		config.data_path = _data_root
		return train_pos, train_neg, val_pos, val_neg
	else:
		return None


def stack_imgfolder(config, label_index, training = True):
	"""
	args:
		<data_root> tree structure:
			class_1:
				**_a.jpg
				**_b.jpg
				**_c.jpg
				++_a.jpg
				.....
			class_2:
				...
	return [train_pos], [train_neg], [val_pos], [val_neg]
	"""
	root       = config.data_path
	suffix_num = len(config.channel_list)
	if training:
		# Process Uncertain folder
		if config.del_uncertain:
			process_uncertain(root)

		train_pos = []
		train_neg = []
		val_pos   = []
		val_neg   = []
		val_rate  = config.validation_rate

		for label in os.listdir(root):
			if label in label_index:
				label_id  = label_index[label]
				label_imgs= []
				img_dir   = os.path.join(root, label)
				all_img   = os.listdir(img_dir)
				valid_imghead = []
				checked_img   = [-1 for _ in range(len(all_img))]
				for img_idx, img in enumerate(all_img):
					if checked_img[img_idx] == 1:
						continue

					suffix = img.split('_')[-1][-5]
					img_head = img[:-6]
					file_suffix = img.split('.')[-1]

					if file_suffix in _valid_suffix:
						other_suffix = set(_suffix[:suffix_num]) - set(suffix)
						_is_valid = True
						for other in other_suffix:
							other_img = img_head + '_' + other + '.' + file_suffix
							if other_img in all_img:
								checked_img[all_img.index(other_img)] = 1
							else:
								_is_valid = False
						if _is_valid:
							label_imgs.append([[os.path.join(img_dir, img_head+ '_' + suf + '.' + file_suffix) for suf in _suffix[:suffix_num]], label_id])
				
				random.shuffle(label_imgs)
				for_val   = label_imgs[:int(val_rate * len(label_imgs))]
				for_train = label_imgs[int(val_rate * len(label_imgs)):]

				if label.lower() == 'ok':
					train_pos.extend(for_train)
					val_pos.extend(for_val)
				else:
					train_neg.extend(for_train)
					val_neg.extend(for_val)

		return train_pos, train_neg, val_pos, val_neg
	else:
		pass
		return []


def ruifeng_by_format(config, label_index, training = True):
	"""说明：当需要做很多组实验时候，可以先生成文件夹，然后进行训练
	args:
		<data_root> tree structure
			1:
				class_1:
					20200507: *.jpg
					20200508: *.jpg
				class_2:
					20200510: *.jpg
				....
			2:  同上,
			3:  同上,
			4:  同上。
		说明：1、2、3、4、5、6、7表示不同的规格，每个规格之下都有包含相同类别集合的图像文件夹，由于要做对比试验因此没有直接
		根据以往的方式进行试验数据的划分，因此，我们可以根据具体的实验setting来有目的性质的划分数据，并写入一个json文件，
	return:
		[train_pos_img], [train_neg_img]; [val_pos_img], [val_neg_img] (Training Mode)
		[all image] (Testing Mode)

	保存文件命名格式为(~1)_(1).json(表示非1文件夹所有图片用于训练，1文件夹所有图片用于测试)
	说明：由于1、2、3、4、5、6、7不同规格的数据差异很大，初步试验发现泛化性能较差。
	"""
	data_root = config.data_path
	if not os.path.exists(data_root):
		print("Root path does not exist, please check your data folder !")
		exit()

	class_name_list = list(label_index.keys())
	if training:
		# 随机挑选一个文件夹作为验证文件夹，其他作为训练，目前如果要更改数据的话，要更改代码
		# val_folder   = random.sample([x for x in range(7)], 1)
		# train_folder = list(set([x for x in range(7)]) - set(val_folder))
		val_folder     = [1]
		train_folder   = [2, 3, 4, 5, 6, 7]
		train_pos = []
		train_neg = []

		for type_idx in train_folder:
			for root, _, files in os.walk(os.path.join(data_root, str(type_idx))):
				for label in class_name_list:
					if label in root:
						label_id = label_index[label]
						if label.lower() == 'ok':
							save_list = train_pos
						else:
							save_list = train_neg

						for img_file in files:
							if '.jpg' in img_file:
								save_list.append((os.path.join(root, img_file), label_id))

		val_pos = []
		val_neg = []
		for type_idx in val_folder:
			for root, _, files in os.walk(os.path.join(data_root, str(type_idx))):
				for label in class_name_list:
					if label in root:
						label_id = label_index[label]
						if label.lower() == 'ok':
							save_list = val_pos
						else:
							save_list = val_neg

						for img_file in files:
							if '.jpg' in img_file:
								save_list.append((os.path.join(root, img_file), label_id))
		return train_pos, train_neg, val_pos, val_neg
	else:
		# 仅提取所有图片和类别信息，不进行类别划分
		all_images = []
		for root, _, files in os.walk(data_root):
			for label in class_name_list:
				if label in root:
					label_id = label_index[label]
					for file in files:
						if '.jpg' in file:
							all_images.append((os.path.join(root, file) , label_id))
		return all_images


def ruifeng_by_date(config, label_index, training = True):
	"""
	args:
		<data_root> tree structure 
			1/
				class_1/
					20200507/ *.jpg
					20200508/ *.jpg
				class_2/
					20200510/ *.jpg
			2/
				...
			...
		说明：1、2、3、4、5、6、7表示不同的规格
	return:
		[train_pos_img], [train_neg_img]; [val_pos_img], [val_neg_img] (Training Mode)
		[all image] (Testing Mode)
	"""
	data_root  = config.data_path
	if training:
		####################################
		#       fetch directory tree       #
		####################################
		_data_tree = {}
		for i in range(1, 8):
			_data_tree[str(i)] = {}
			id_subdir = _data_tree[str(i)]
			for root, _, files in os.walk(data_root + '/' +str(i)):
				if len(files) != 0:
					if 'uncertain' not in root:
						# root_path_list = root.split('/')[-1].split('\\') # windows-version
						root_path_list = root.split('/')
						label = root_path_list[-2]
						date  = root_path_list[-1]
						if label in id_subdir:
							id_subdir[label][date] = files
						else:
							id_subdir[label] = {date: files}

		###############################################
		#      Collect date_set for each sub-dir      #
		###############################################
		date_set = []
		for one_dir in _data_tree.values():
			date_one_dir = []
			for v in one_dir.values():
				date_one_dir.extend(list(v.keys()))

			date_one_dir = list(set(date_one_dir))
			date_one_dir.sort()
			date_set.append(date_one_dir)

		#####################################################
		#    Split dataset for train and eval w.r.t date    #
		#####################################################
		date_train_list   = []
		date_train_sel_id = []
		for date_all, subdir_tree in zip(date_set, _data_tree.values()):
			# Count image numbers for each date
			date_img_num = [0] * len(date_all)
			for class_d_img in subdir_tree.values():
				for date, file in class_d_img.items():
					date_img_num[date_all.index(date)] += len(file)

			sample_num    = len(date_all)
			min_gap       = 1
			sel_gp        = None
			success_split = False
			while sample_num > 1:
				sample_num -= 1
				for group in combine(len(date_all), sample_num):
					gp_ratio = sum([date_img_num[x] for x in group]) / sum(date_img_num)
					if gp_ratio>=0.65 and gp_ratio<=0.7:
						sel_gp = group
						success_split = True
						final_ratio   = gp_ratio
						break
					elif min(abs(gp_ratio - 0.65), abs(gp_ratio - 0.7)) < min_gap:
							final_ratio = gp_ratio
							sel_gp = group
							min_gap = min(abs(gp_ratio - 0.65), abs(gp_ratio - 0.7))
				if success_split:
					break

			date_train_sel_id.append(sel_gp)
			date_train_list.append([date_all[x] for x in sel_gp])
		# split_md5 = str(date_train_sel_id)

		###############################################
		#              final split                    #
		###############################################
		train_pos = []
		train_neg = []
		val_pos   = []
		val_neg   = []
		for id_dir, (date_train, subdir_tree) in enumerate(zip(date_train_list, _data_tree.values()), start = 1):
			for label, date_2_file in subdir_tree.items():
				label_id = label_index[label]
				for date, img_files in date_2_file.items():
					if date in date_train:
						if label.lower() == 'ok':
							for file in img_files:
								train_pos.append([os.path.join(data_root, str(id_dir), label, date, file), label_id])
						else:
							for file in img_files:
								train_neg.append([os.path.join(data_root, str(id_dir), label, date, file), label_id])
					else:
						if label.lower() == 'ok':
							for file in img_files:
								val_pos.append([os.path.join(data_root, str(id_dir), label, date, file), label_id])
						else:
							for file in img_files:
								val_neg.append([os.path.join(data_root, str(id_dir), label, date, file), label_id])
		return train_pos, train_neg, val_pos, val_neg
	else:
		# To be Continued
		return None


def ruifeng_by_class(config, label_index, training = True):
	"""
	args:
		<data_root> tree structure 
			1/
				class_1/
					20200507/ *.jpg
					20200508/ *.jpg
				class_2/
					20200510/ *.jpg
			2/
				...
			...
		说明：1、2、3、4、5、6、7表示不同的规格
	return:
		[train_pos_img], [train_neg_img]; [val_pos_img], [val_neg_img] (Training Mode)
		[all image] (Testing Mode)
	"""
	pass