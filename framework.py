import numpy as np
import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import math
import pandas as pd

import utils
import filters.base as filter
import metrics 
import algorithms.base as base

# from method_filter import rasism_list

from IPython.display import display

class frame:
    '''
    фреймворк

    Аргументы:
       -color: тип загрузки изображений цветное/черно-белое
       -folder_path: директория с исходными изображениями
       -folder_path_blured: директория со смазанными изображениями
       -folder_path_restored: директория с восстановленными изображениями
       -images: массив связей изображений с их смазанными и восстановленными версиями
       -data_path: директория, куда сохранять анализ данных
    '''
    color = False #grayscale/color
    folder_path = 'images'
    folder_path_blured = 'blured'
    folder_path_restored = 'restored'
    images = np.array([])

    def __init__(self, images_folder = 'images', blured_folder = 'blured', restored_folder = 'restored', data_path = 'data'):
        '''
        фреймворк и его интерфейс
        создаст директории при их отсутствии
        Аргументы:
            -images_folder: путь директории оригинальных изображений
            -blured_folder: путь директории смазанных изображений
            -restored_folder: путь директори восстановленных изображений
            -data_path: директория, куда сохранять анализ данных
        '''
        self.folder_path = images_folder
        self.folder_path_blured = blured_folder
        self.folder_path_restored = restored_folder
        self.data_path = data_path
        self.amount_of_blurred = 1
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
        if not os.path.exists(blured_folder):
            os.makedirs(blured_folder)
        if not os.path.exists(restored_folder):
            os.makedirs(restored_folder)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    def changescale(self, color: bool):
        '''
        Меняет способ загрузки новых изображений
        Аргументы:
            - color: True - цветное, False - черно-белое
        '''
        self.color = color

    def read_all(self):
        '''
        загружает все изображения из директории в фреймворк
        '''
        image_files = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]
        for image_file in image_files:
            image_path = os.path.join(self.folder_path, image_file)
            
            if self.color:
                image = cv.imread(image_path, cv.IMREAD_COLOR)
            else:
                image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            
            # кириллицу метод не читает
            # не картинки вроде можно игнорить
            if image is not None:
                self.images = np.append(self.images, utils.Image(image_path,self.color))

    def read_one(self, path):
        '''
        загружает из директории одно изображение в фреймворк с заданным названием

        Аргументы:
            -path: название файла
        '''
        image_path = os.path.join(self.folder_path, path)

        if self.color:
            image = cv.imread(image_path, cv.IMREAD_COLOR)
        else:
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        if image is not None:
            self.images = np.append(self.images, utils.Image(image_path,self.color))

    def show_original(self):
        '''
        выводит все оригинальные изображения через cv.imshow
        '''
        for i in self.images:
            img = i.get_original_image()
            if img is not None:
                cv.imshow("original",img)
                cv.waitKey(0)
                cv.destroyAllWindows()
            print(i.get_original())
    
    def show_blured(self):
        '''
        выводит все смазанные изображения через cv.imshow
        '''
        for i in self.images:
            img = i.get_blured_image()
            if img is not None:
                cv.imshow("blured",img)
                cv.waitKey(0)
                cv.destroyAllWindows()

    def show_restored(self):
        '''
        выводит все восстановленные изображения через cv.imshow
        '''
        for i in self.images:
            for j in zip(i.get_algorithm(),i.get_restored_image()):
                img = j[1]
                if img is not None:
                    cv.imshow(j[0],j[1])
                    cv.waitKey(0)
                    cv.destroyAllWindows()

    def filter(self, filter_processor: filter.FilterBase):
        '''
        применяет фильтр ко всем загруженным изображениям
        сохраняет изменения в blured_folder

        Аргументы:
            -filter_processor: фильтр
        '''
        for img in self.images:
            if img.get_blured() is not None:
                original = img.get_blured_image()
            else:
                original = img.get_original_image()
    
            if original is None:
                raise Exception("failure to open image")
            
            filtered_image = filter_processor.filter(original)
            if img.get_blured() is None:
                original_filename = os.path.basename(img.get_original())

                base_name = f"{original_filename}"
                new_path = os.path.join(self.folder_path_blured, base_name)
                file_counter = 1

                while os.path.exists(new_path):
                    name, ext = os.path.splitext(base_name)
                    new_path = os.path.join(self.folder_path_blured, f"{name}_{file_counter}{ext}")
                    file_counter += 1
            else:
                original_filename = os.path.basename(img.get_blured())
                base_name = f"{original_filename}"
                new_path = os.path.join(self.folder_path_blured, base_name)
    
            cv.imwrite(new_path,filtered_image)
            img.set_blured(new_path)
            img.add_to_current_filter(filter_processor.discription())
    
    def show(self, size:float = 1.0):
        '''
        выводит изображения следующим образом:
        оригинам | смазанное + метрики | восстановленное + метрики ... 
        Аргументы:
            -size: коэф. размера таблицы (чисто визуальный параметр)
        '''
        for img in self.images:
            self.show_with_metrics("Image Restoration Metrics Comparison",img,size)
        pass

    def show_with_metrics(self, title, img, size:float = 1.0):
            '''
            выводит изображение следующим образом:
            оригинам | смазанное + метрики | восстановленное + метрики ... 
            Аргументы:
                -title: заголовок таблицы
                -img: связь изображений, которую надо отобразить
                -size: коэф. размера таблицы (чисто визуальный параметр)
            '''
            bool_blur = int(img.get_blured() is not None)
            fig, axes = plt.subplots(1, 1+bool_blur+len(img.get_restored()), figsize=((5+5*bool_blur+5*len(img.get_restored()))*size, 4*size))
            if len(img.get_restored()) == 0 and bool_blur == 0:
                axes.imshow(cv.cvtColor(img.get_original_image(),cv.COLOR_BGR2RGB))
                axes.set_title("Original", fontsize=12)
                axes.axis('off')
                plt.suptitle(title, y=1.02, fontsize=14*size)
                plt.tight_layout()
                plt.show()
                return
            else:
                axes[0].imshow(cv.cvtColor(img.get_original_image(),cv.COLOR_BGR2RGB))
                axes[0].set_title("Original", fontsize=12)
                axes[0].axis('off')
            if(img.get_blured() is not None):
                blured_img = img.get_blured_image()
                blured_psnr = metrics.PSNR(img.get_original_image(),blured_img)
                try:
                    blured_ssim = metrics.SSIM(img.get_original_image(),blured_img)
                except:
                    blured_ssim = math.nan
                axes[1].imshow(cv.cvtColor(blured_img,cv.COLOR_BGR2RGB))
                axes[1].set_title(f"Blured\nPSNR: {blured_psnr:.4f} | SSIM: {blured_ssim:.4f}", fontsize=10)
                axes[1].axis('off')

            for i in range(1+bool_blur, axes.shape[0]):
                axes[i].axis('off')
            
            restored_images = img.get_restored_image()
            PSNR = img.get_PSNR()
            SSIM = img.get_SSIM()
            ALG = img.get_algorithm()

            for i, alg_iter in enumerate(ALG):
                
                image_iter = restored_images[i]
                restored_img = image_iter
                psnr = PSNR[i]
                ssim = SSIM[i]
                alg = alg_iter

                ax = axes[i+bool_blur+1]
                ax.imshow(cv.cvtColor(restored_img,cv.COLOR_BGR2RGB))
                ax.set_title(f"{alg}\nPSNR: {psnr:.4f} | SSIM: {ssim:.4f}", fontsize=10)
                ax.axis('off')
            plt.suptitle(title, y=1.02, fontsize=14*size)
            plt.tight_layout()
            plt.show()
            
    def get_metrics(self):
        '''
        Выходные данные:
            возвращает метрики всех восстановленных изображений в качестве массива
        '''
        metrics = list()
        for i in self.images:
            metrics.append((i.get_PSNR(), i.get_SSIM()))
        return metrics

    def clear_input(self):
        '''
        убирает привязку ко всем загруженным изображениям (загружать надо заново)
        '''
        self.images = np.array([])
    
    def reset(self):
        '''
        убирает привязку к отфильтрованным и восстановленным изображениями
        '''
        for i in self.images:
            i.set_blured(None)
            i.set_restored(list())
            i.set_PSNR(np.array([]))
            i.set_SSIM(np.array([]))
            i.set_algorithm(np.array([]))

            i.set_blurred_array(np.array([]))
            i.set_current_filter(None)
            i.set_filters(np.array([]))
        
    def clear_output(self):
        '''
        удаляет все привязанные отфильрованные и восстановленные изображениями
        '''
        for i in self.images:
            tmpfile = i.get_blured()
            if tmpfile is not None:
                os.remove(tmpfile)
            for j in i.get_restored():
                if j is not None:
                    os.remove(j)
            for tmpfile in i.get_blurred_array():
                if tmpfile is not None:
                    os.remove(tmpfile)

        self.reset()

    def clear_output_directory(self, WARNING = 'IT WILL DELETE EVERYTHING!'):
        '''
        УДАЛЯЕТ ВСЕ из директорий с отфильрованными и восстановленными изображениями
        '''
        security = input(f"YOU SURE, YOU WANT TO DELETE EVERY SINGLE FILE IN DIRECTORIES {self.folder_path_blured} AND {self.folder_path_restored}? (YES)")
        if security != 'YES':
            print("operation canceled")
            return
        folder_path = self.folder_path_blured

        if not os.path.exists(folder_path):
            raise Exception('could not find blured folder')
        

        files = glob.glob(os.path.join(folder_path,'*'))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"could not delete{f}: {e}")

        folder_path = self.folder_path_restored

        if not os.path.exists(folder_path):
            raise Exception('could not find blured folder')
        

        files = glob.glob(os.path.join(folder_path,'*'))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"could not delete{f}: {e}")
        self.reset()  
    
    def bind(self, original_image_path, blured_image_path, color:bool = True):
        '''
        добавляет связь
        Аргументы:
            -original_image_path: путь к оригинальному изображению (полностью)
            -blured_image_path: путь к смазанному изображению (полностью)
            -color: способ загрузки (цветное/черно-белое)
        '''
        tmp1 = cv.imread(original_image_path, cv.IMREAD_COLOR if color else cv.IMREAD_GRAYSCALE)
        tmp2 = cv.imread(blured_image_path, cv.IMREAD_COLOR if color else cv.IMREAD_GRAYSCALE)
        if tmp1 is None:
            raise Exception(f"unable to open image {tmp1}")
        if tmp2 is None:
            raise Exception(f"unable to open image {tmp2}")
        img = utils.Image(original_image_path,color)
        img.set_blured(blured_image_path)
        img.set_current_filter("unknown")
        self.images= np.append(self.images,img)
    
    def process(self, algorithm_processor: base.DeconvolutionAlgorithm):
        '''
        восстановление всех изображений
        Аргументы:
            -algorithm_processor: метод восстановления изображения
        '''
        alg_name = algorithm_processor.get_name()

        for i in self.images:
            original_image = i.get_original_image()
            blured_image = i.get_blured_image()
            if blured_image is None:
                blured_image = original_image
            restored_image = algorithm_processor.process(blured_image)

            if i.get_blured() is not None:
                blured_image_path = os.path.basename(i.get_blured())
            else:
                blured_image_path = os.path.basename(i.get_original())

            blured_image_path_name, ext = os.path.splitext(blured_image_path)
            restored_path = os.path.join(self.folder_path_restored,f"{blured_image_path_name}_{alg_name}{ext}")
            file_counter = 1
            
            while os.path.exists(restored_path):
                name2, ext2 = os.path.splitext(os.path.basename(restored_path))
                restored_path = os.path.join(self.folder_path_restored, f"{name2}_{file_counter}{ext2}")
                file_counter += 1
            
            cv.imwrite(restored_path,restored_image)

            i.add_PSNR(metrics.PSNR(original_image,restored_image))
            try:
                i.add_SSIM(metrics.SSIM(original_image,restored_image)) #must be odd =/ + L + ratio
            except:
                i.add_SSIM(math.nan)
            i.add_algorithm(alg_name)
            i.add_restored(restored_path)
    
    def show_all(self, size: float = 1.0):
        print("Warning: show_all only work after full_process!")
        '''
            выводит изображение следующим образом:
            оригинам | смазанное + метрики | восстановленное + метрики ... 
            достает изображения из списка, выводит как таблицу
            Аргументы:
                -size: коэф. размера таблицы (чисто визуальный параметр)
        '''
        h = 0
        w = self.images[0].get_len_algorithms()+2
        line = 0
        for i in self.images:
            h += i.get_len_filter()+1
        fig, axes = plt.subplots(h, w, figsize=((5*w*size, 4*h*size)))

        for img in self.images:
            cur_h = img.get_len_filter()+1
            cur_w = w - 2
            cur_line = 0
            # print(f"h: {h}, w: {w}, line: {line}, cur_h: {cur_h}, cur_w: {cur_w}, cur_line: {cur_line}, alg: {len(img.get_algorithm())}")
            restored = img.get_restored().reshape(cur_h,cur_w)
            psnr = img.get_PSNR().reshape(cur_h,cur_w)
            ssim = img.get_SSIM().reshape(cur_h,cur_w)
            blurred_array = img.get_blurred_array()

            #blurred_path
            axes[line, 0].imshow(cv.cvtColor(img.get_original_image(),cv.COLOR_BGR2RGB))
            axes[line, 0].set_title("Original", fontsize=12)
            axes[line, 0].axis('off')

            blured_img = img.get_blured_image()
            try:
                blured_psnr = metrics.PSNR(img.get_original_image(),blured_img)
            except:
                blured_psnr = math.nan
            try:
                blured_ssim = metrics.SSIM(img.get_original_image(),blured_img)
            except:
                blured_ssim = math.nan
            axes[line,1].imshow(cv.cvtColor(blured_img,cv.COLOR_BGR2RGB))
            axes[line,1].set_title(f"Blured\nPSNR: {blured_psnr:.4f} | SSIM: {blured_ssim:.4f}", fontsize=10)
            axes[line,1].axis('off')
            for i in range(2, axes.shape[1]):
                axes[line,i].axis('off')
            
            restored_images = restored[0,:]
            PSNR = psnr[0,:]
            SSIM = ssim[0,:]
            ALG = img.get_algorithm()

            for i, image_iter in enumerate(restored_images):

                _restored_img = cv.imread(image_iter,cv.IMREAD_COLOR if img.get_color() else cv.IMREAD_GRAYSCALE)
                _psnr = PSNR[i]
                _ssim = SSIM[i]
                _alg = ALG[i]

                ax = axes[line,i+2]
                ax.imshow(cv.cvtColor(_restored_img,cv.COLOR_BGR2RGB))
                ax.set_title(f"{_alg}\nPSNR: {_psnr:.4f} | SSIM: {_ssim:.4f}", fontsize=10)
                ax.axis('off')
            
            #all other
            line+=1
            for cur_line in range(0,cur_h-1):
                axes[line, 0].imshow(cv.cvtColor(img.get_original_image(),cv.COLOR_BGR2RGB))
                axes[line, 0].set_title("Original", fontsize=12)
                axes[line, 0].axis('off')

                blured_img = cv.imread(blurred_array[cur_line],cv.IMREAD_COLOR if img.get_color() else cv.IMREAD_GRAYSCALE)
                try:
                    blured_psnr = metrics.PSNR(img.get_original_image(),blured_img)
                except:
                    blured_psnr = math.nan
                try:
                    blured_ssim = metrics.SSIM(img.get_original_image(),blured_img)
                except:
                    blured_ssim = math.nan
                axes[line,1].imshow(cv.cvtColor(blured_img,cv.COLOR_BGR2RGB))
                axes[line,1].set_title(f"Blured\nPSNR: {blured_psnr:.4f} | SSIM: {blured_ssim:.4f}", fontsize=10)
                axes[line,1].axis('off')


                for i in range(2, axes.shape[1]):
                    axes[line,i].axis('off')
            
                restored_images = restored[cur_line+1,:]

                PSNR = psnr[cur_line+1,:]
                SSIM = ssim[cur_line+1,:]
                # ALG = img.get_algorithm()

                for i, image_iter in enumerate(restored_images):

                    _restored_img = cv.imread(image_iter,cv.IMREAD_COLOR if img.get_color() else cv.IMREAD_GRAYSCALE)
                    _psnr = PSNR[i]
                    _ssim = SSIM[i]
                    _alg = ALG[i]

                    ax = axes[line,i+2]
                    ax.imshow(cv.cvtColor(_restored_img,cv.COLOR_BGR2RGB))
                    ax.set_title(f"{_alg}\nPSNR: {_psnr:.4f} | SSIM: {_ssim:.4f}", fontsize=10)
                    ax.axis('off')
                line+=1
                pass
        plt.suptitle("ANALITICS", y=1.02, fontsize=14*size)
        plt.tight_layout()
        plt.show()

        pass

    def save_filter(self):
        '''
        переносит изображение из буфера в список
        к изображениям в списке не применяются фильтры
        '''
        for i in self.images:
            i.save_filter()
        self.amount_of_blurred = self.amount_of_blurred + 1

    def load_filter(self, index):
        '''
        достает изображение из списка в буфер, для изменения
        Аргументы:
            -index: индекс доставаемого изображения
        '''
        for i in self.images:
            i.load(index)
        self.amount_of_blurred = self.amount_of_blurred - 1

    def len_blur(self):
        '''
        возвращает количество смазанных вариантов одного изображения
        '''
        return self.amount_of_blurred

    def full_process(self, filters: np.array, methods: np.array, size: float = 0.75):
        '''
        пайплайн применения смазов с последующим восстановлением и анализов результатов
        Аргументы:
            -filters: массив массивов объектов класса FilterBase (фильтры к изображению) [[],[]]
            -methods: массив объектов класса DeconvolutionAlgorithm (методы восстановления) []
            -size: размер таблицы лучших/худших метрик
        '''

        df_psnr = pd.DataFrame()
        df_ssim = pd.DataFrame()

        df_images = pd.DataFrame()
        images_dict = {}
        ssim_dict = {}
        psnr_dict = {}
        for img in self.images:
            for filters_iterations in filters:
                img.save_filter()
                original_image = img.get_original_image()

                if img.get_blured() is not None:
                    filtered_image = img.get_blured_image()
                else:
                    filtered_image = original_image
        
                if filtered_image is None:
                    raise Exception("failure to open image")
            
            
                for filter_processor in filters_iterations:
                    filtered_image = filter_processor.filter(filtered_image)
                    img.add_to_current_filter(filter_processor.discription())

                if img.get_blured() is None:
                    original_filename = os.path.basename(img.get_original())

                    base_name = f"{original_filename}"
                    new_path = os.path.join(self.folder_path_blured, base_name)
                    file_counter = 1

                    while os.path.exists(new_path):
                        name, ext = os.path.splitext(base_name)
                        new_path = os.path.join(self.folder_path_blured, f"{name}_{file_counter}{ext}")
                        file_counter += 1
                else:
                    original_filename = os.path.basename(img.get_blured())
                    base_name = f"{original_filename}"
                    new_path = os.path.join(self.folder_path_blured, base_name)
        
                cv.imwrite(new_path,filtered_image)
                img.set_blured(new_path)

                try:
                    psnr_tmp_blurred = metrics.PSNR(original_image,filtered_image)
                except:
                    psnr_tmp_blurred = math.nan

                try:
                    ssim_tmp_blurred = metrics.SSIM(original_image,filtered_image)
                except:
                    ssim_tmp_blurred = math.nan

                images_dict['original'] = np.append(img.get_original(), images_dict.setdefault('original',[]))
                psnr_dict['original'] = np.append(img.get_original(), psnr_dict.setdefault('original',[]))
                ssim_dict['original'] = np.append(img.get_original(), ssim_dict.setdefault('original',[]))

                images_dict['filter'] = np.append(img.get_current_filter(),images_dict.setdefault('filter',[]))
                psnr_dict['filter'] = np.append(img.get_current_filter(),psnr_dict.setdefault('filter',[]))
                ssim_dict['filter'] = np.append(img.get_current_filter(),ssim_dict.setdefault('filter',[]))

                images_dict['blurred'] = np.append(new_path, images_dict.setdefault('blurred',[]))
                psnr_dict['blurred'] = np.append(psnr_tmp_blurred, psnr_dict.setdefault('blurred',[]))
                ssim_dict['blurred'] = np.append(ssim_tmp_blurred, ssim_dict.setdefault('blurred',[]))

            
            for algorithm_processor in methods:

                alg_name = algorithm_processor.get_name()

                blured_image = filtered_image
                restored_image = algorithm_processor.process(blured_image)

                if img.get_blured() is not None:
                    blured_image_path = os.path.basename(img.get_blured())
                else:
                    blured_image_path = os.path.basename(img.get_original())

                blured_image_path_name, ext = os.path.splitext(blured_image_path)
                # print(blured_image_path_name)
                restored_path = os.path.join(self.folder_path_restored,f"{blured_image_path_name}_{alg_name}{ext}")
                file_counter = 1
            
                while os.path.exists(restored_path):
                    name2, ext2 = os.path.splitext(os.path.basename(restored_path))
                    restored_path = os.path.join(self.folder_path_restored, f"{name2}_{file_counter}{ext2}")
                    file_counter += 1
            
                cv.imwrite(restored_path,restored_image)
                try:
                    restored_psnr = metrics.PSNR(original_image,restored_image)
                except:
                    restored_psnr = math.nan
                img.add_PSNR(restored_psnr)
                try:
                    restored_ssim = metrics.SSIM(original_image,restored_image)#must be odd =/ + L + ratio
                except:
                    restored_ssim = math.nan
                img.add_SSIM(restored_ssim)

                img.add_algorithm(alg_name)
                img.add_restored(restored_path)

                images_dict[alg_name] = np.append(restored_path,images_dict.setdefault(alg_name,[]))
                psnr_dict[alg_name] = np.append(restored_psnr,psnr_dict.setdefault(alg_name,[]))
                ssim_dict[alg_name] = np.append(restored_ssim,ssim_dict.setdefault(alg_name,[]))

            # for algorithm_processor in methods:
            #     img.add_algorithm(algorithm_processor.get_name())
            # print(img.get_blured())
            # print(img.get_blurred_array())
            for blurred_array_iter in img.get_blurred_array():
                blured_image = cv.imread(blurred_array_iter,cv.IMREAD_COLOR if img.get_color() else cv.IMREAD_GRAYSCALE)
                
                for algorithm_processor in methods:
                    alg_name = algorithm_processor.get_name()

                    restored_image = algorithm_processor.process(blured_image)

                    blured_image_path = os.path.basename(blurred_array_iter)

                    blured_image_path_name, ext = os.path.splitext(blured_image_path)
                    # print(blured_image_path_name)
                    restored_path = os.path.join(self.folder_path_restored,f"{blured_image_path_name}_{alg_name}{ext}")
                    file_counter = 1
                
                    while os.path.exists(restored_path):
                        name2, ext2 = os.path.splitext(os.path.basename(restored_path))
                        restored_path = os.path.join(self.folder_path_restored, f"{name2}_{file_counter}{ext2}")
                        file_counter += 1
                
                    cv.imwrite(restored_path,restored_image)
                    try:
                        restored_psnr = metrics.PSNR(original_image,restored_image)
                    except:
                        restored_psnr = math.nan
                    img.add_PSNR(restored_psnr)
                    try:
                        restored_ssim = metrics.SSIM(original_image,restored_image)#must be odd =/ + L + ratio
                    except:
                        restored_ssim = math.nan
                    img.add_SSIM(restored_ssim)
                    # img.add_algorithm(alg_name)
                    img.add_restored(restored_path)

                    images_dict[alg_name] = np.append(restored_path,images_dict.setdefault(alg_name,[]))
                    psnr_dict[alg_name] = np.append(restored_psnr,psnr_dict.setdefault(alg_name,[]))
                    ssim_dict[alg_name] = np.append(restored_ssim,ssim_dict.setdefault(alg_name,[]))
        
        self.show_all(size)
        df_images = pd.DataFrame(images_dict)
        df_psnr = pd.DataFrame(psnr_dict)
        df_ssim = pd.DataFrame(ssim_dict)
        display(df_images)
        display(df_psnr)
        display(df_ssim)
        # df_images.to_excel(f"{self.data_path}\\image_df.xlsx")
        # df_psnr.to_excel(f"{self.data_path}\\psnr_df.xlsx")
        # df_ssim.to_excel(f"{self.data_path}\\ssim_df.xlsx")
        df_images.to_csv(f"{self.data_path}\\image_df.csv",index=False)
        df_psnr.to_csv(f"{self.data_path}\\psnr_df.csv",index=False)
        df_ssim.to_csv(f"{self.data_path}\\ssim_df.csv",index=False)
        self.analise(df_images, df_psnr, df_ssim)

        pass

    def analise(self, df_images, df_psnr,df_ssim):
        '''
        do some shit to process this fucking stupid data...
        i hate my life
        '''
        pass



        


