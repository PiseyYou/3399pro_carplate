import numpy as np
import cv2
import sys
import os
from rknn.api import RKNN
import random
import threading
import multiprocessing
# import matplotlib.pyplot as plt
from time import time
import math
import shutil

platechar_dic = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '10': 'A',
    '11': 'B', '12': 'C', '13': 'D', '14': 'E', '15': 'F',
    '16': 'G', '17': 'H', '18': 'J', '19': 'K', '20': 'L', '21': 'M', '22': 'N', '23': 'P', '24': 'Q', '25': 'R',
    '26': 'S', '27': 'T', '28': 'U', '29': 'V', '30': 'W', '31': 'X', '32': 'Y', '33': 'Z',

    "34": '澳', "35": '川', "36": '桂', "37": '贵', "38": '冀', "39": '吉', "40": '京', "41": '琼', "42": '陕', "43": '苏',
    "44": '湘', "45": '豫', "46": '渝',
    "47": '鄂', "48": '甘', "49": '赣', "50": '港', "51": '黑', "52": '沪', "53": '晋', "54": '津', "55": '鲁', "56": '蒙',
    "57": '闽', "58": '宁',
    "59": '青', "60": '使', "61": '皖', "62": '新', "63": '学', "64": '云', "65": '藏', "66": '浙', "67": '警', '68': '粤',
    '69': '辽', '70': '应', '71': '民',

}


class RECT:
    def __init__(self, ltx, lty, rbx, rby):
        self.P_ltx = ltx
        self.P_lty = lty
        self.P_rbx = rbx
        self.P_rby = rby
        self.area = (rbx - ltx) * (rby - lty)


def fixEdge_H(p_val, img_h):
    if p_val <= 0:
        p_val = 1
    if p_val >= img_h:
        p_val = img_h - 1
    return p_val


def fixEdge_W(p_val, img_w):
    if p_val <= 0:
        p_val = 1
    if p_val >= img_w:
        p_val = img_w - 1
    return p_val


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


class plate_pro:

    def __init__(self):
        rknnModel_dir = './model_0107'
        rknnModel_name = os.path.join(rknnModel_dir, 'vehicle_DetRkmodel1231_qu.rknn')
        self.rknn_vehicleDet = RKNN(verbose=False)
        ret = self.rknn_vehicleDet.load_rknn(path=rknnModel_name)
        ret = self.rknn_vehicleDet.init_runtime()
        self.VEHICLE_NUM_RESULTS = 294  # 207#702#207
        self.VEHICLE_NUM_CLASS = 2
        priorbox_name = os.path.join(rknnModel_dir, 'Vehicle_DetBox_1219.txt')
        priorbox = []
        with open(priorbox_name) as f:
            for line in f:
                data = line.strip()
                priorbox.append(data)
        priorbox = np.array(priorbox, np.float16)
        box_priors = priorbox.reshape((self.VEHICLE_NUM_RESULTS, 4))
        self.vehicle_priorbox = box_priors * 0.5

        rknnModel_name = os.path.join(rknnModel_dir, 'plate_DetRkmodel_qu1227.rknn')
        self.rknn_PlateDet = RKNN(verbose=False)
        ret = self.rknn_PlateDet.load_rknn(path=rknnModel_name)
        ret = self.rknn_PlateDet.init_runtime()
        self.PLATE_NUM_RESULTS = 2500  # 3750#1250#3174
        self.PLATE_NUM_CLASS = 2
        priorbox_name = os.path.join(rknnModel_dir, 'Plate_DetBox1224.txt')
        priorbox = []
        with open(priorbox_name) as f:
            for line in f:
                data = line.strip()
                priorbox.append(data)
        priorbox = np.array(priorbox, np.float16)
        box_priors = priorbox.reshape((self.PLATE_NUM_RESULTS, 4))
        self.plate_priorbox = box_priors * 0.5

        rknnModel_name = os.path.join(rknnModel_dir, 'platechar_DetRkmodel_qu0106.rknn')
        self.rknn_PlateCharDet = RKNN(verbose=False)
        ret = self.rknn_PlateCharDet.load_rknn(path=rknnModel_name)
        ret = self.rknn_PlateCharDet.init_runtime()
        self.PLATECHAR_NUM_RESULTS = 2500  # 1250 #5000
        self.PLATECHAR_NUM_CLASS = 2
        priorbox_name = os.path.join(rknnModel_dir, 'Platechar_DetBox0106.txt')
        priorbox = []
        with open(priorbox_name) as f:
            for line in f:
                data = line.strip()
                priorbox.append(data)
        priorbox = np.array(priorbox, np.float16)
        box_priors = priorbox.reshape((self.PLATECHAR_NUM_RESULTS, 4))
        self.platechar_priorbox = box_priors * 0.5

        rknnModel_name = os.path.join(rknnModel_dir, 'charrecog0102.rknn')
        self.rknn_PlateCharRecog = RKNN(verbose=False)
        ret = self.rknn_PlateCharRecog.load_rknn(path=rknnModel_name)
        ret = self.rknn_PlateCharRecog.init_runtime(target='rk1808')

        self.Char_avgH = 0
        self.Char_avgW = 0
        self.Char_name = np.zeros(10, np.int)
        self.prob_arr = np.zeros((7, 12), np.int)
        self.Color = '蓝'
        self.nCnChar_Count = 0
        self.nEnChar_Count = 0
        self.layer_type = 'single'
        self.plate_type = ''

        # track params
        self.rect_num = 0
        self.rect_preboxes = np.zeros([self.rect_num, 8],
                                      np.int)  # left,top,right,bottom,cal_iou_sy,recog_num,track_num,ap_max
        self.rect_aftboxes = np.zeros([self.rect_num, 8], np.int)
        self.track_miss = 0
        self.plate_state = 0
        self.track_conut = 0
        self.none_plate_ct = 0
        self.vehicle_conut = 0
        self.plate_result = []
        self.ap_max = 0
        self.plate_max = []
        self.plate_result_lock = -1
        self.last_result = ''
        self.saveDif_min = 1000

        self.type_max_num = 0
        self.type_list_type = []
        self.type_max_type = ''

        # save_result
        # save_result = './result.txt'
        # self.f = open(save_result,'w')

    # def __del__(self):
    #    self.rknn_vehicleDet.release()
    #    self.rknn_PlateDet.release()
    #    self.rknn_PlateCharDet.release()
    #    self.rknn_PlateCharRecog.release()
    #    self.f.close()

    def preprocess(self, src):
        img = cv2.resize(src, (180, 64))
        img = img - 127.5
        img = img * 0.007843
        return img

    def preprocess_vehicle_det(self, src):
        img = cv2.resize(src, (100, 100))
        return img

    def preprocess_plate_det(self, src):
        if len(np.shape(src)) is 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(src, (200, 200))

        return img

    def preprocess_platechar_det(self, src):
        if len(np.shape(src)) is 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(src, (100, 100))

        return img

    def preprocess_platechar_recog(self, src):
        if len(np.shape(src)) is 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        src = cv2.resize(src, (24, 24))
        return src

    def postprocess(self, img, out):
        h = img.shape[0]
        w = img.shape[1]
        box = out[:, 3:7] * np.array(
            [w, h, w, h])  # np.array([180, 64, 180, 64])#* np.array([w, h, w, h])

        cls = out[:, 1]
        conf = out[:, 2]
        return (box.astype(np.int32), conf, cls)

    def draw_rect(self, boxes, img):
        for box in boxes:
            p1 = (box[0], box[1])
            p2 = (box[2], box[3])
            cv2.rectangle(img, p1, p2, (0, 0, 255), 2)

        return 0

    def draw_rect_track(self, boxes, img, track_num):
        for box in boxes:
            p1 = (box[0], box[1])
            p2 = (box[2], box[3])
            cv2.rectangle(img, p1, p2, (0, 0, track_num * 5), 10)

        return 0

    def saveTest_result(self, save_dir, recog_result):
        if recog_result:
            right_save = os.path.join(save_dir, 'right/')
            if not os.path.exists(right_save):
                os.makedirs(right_save)
            cv2.imwrite(right_save + img_name, img_src)
        else:
            wrong_save = os.path.join(save_dir, 'wrong/')
            if not os.path.exists(wrong_save):
                os.makedirs(wrong_save)
            cv2.imwrite(wrong_save + img_name, img_src)

    def judge_platebox_bef(self, box):

        rect_w = box[2] - box[0]
        rect_h = box[3] - box[1]

        rat = rect_w / rect_h

        return False

    def cal_IOU(self, rect_1, rect_2):

        W = min(rect_1.P_rbx, rect_2.P_rbx) - max(rect_1.P_ltx, rect_2.P_ltx)

        H = min(rect_1.P_rby, rect_2.P_rby) - max(rect_1.P_lty, rect_2.P_lty)

        if W <= 0 or H <= 0:
            IOU = 0
            cross = 0
        else:
            area_1 = (rect_1.P_rby - rect_1.P_lty) * (rect_1.P_rbx - rect_1.P_ltx)
            area_2 = (rect_2.P_rby - rect_2.P_lty) * (rect_2.P_rbx - rect_2.P_ltx)
            cross = W * H
            IOU = cross / (area_1 + area_2 - cross)

        return IOU, cross

    def chooseScroe(self, box, conf, cls, threshold):
        box_result = []
        cls_result = []
        det_sy = np.zeros(len(box))
        qe = -1
        for i in range(len(box)):
            if det_sy[i] == -2 or (box[i][0] < 0 and box[i][1] < 0 and box[i][2] < 0 and box[i][3] < 0):
                continue
            sy = 0
            p1 = [box[i][0], box[i][1]]
            p2 = [box[i][2], box[i][3]]

            Rect_1 = RECT(p1[0], p1[1], p2[0], p2[1])

            for j in range(len(box)):
                if j is i or det_sy[j] is 1:
                    continue
                p3 = [box[j][0], box[j][1]]
                p4 = [box[j][2], box[j][3]]

                if p3[0] > p2[0] or p4[0] < p1[0]:  # junmp not crossing
                    continue

                Rect_2 = RECT(p3[0], p3[1], p4[0], p4[1])

                IOU, cross = self.cal_IOU(Rect_1, Rect_2)

                if IOU > threshold and conf[i] >= conf[j]:
                    det_sy[j] = -2

                if IOU > threshold and conf[i] < conf[j]:  # get maxscore rect in IOU>threshold
                    sy = 1
                    det_sy[i] = 1
                    break

                # if cross / Rect_1.area > 0.5 and Rect_1.area < Rect_2.area and cross  / Rect_2.area < 0.5:  # dis small rect
                #   sy = 1
                #  det_sy[i] = 1
                # break
            if sy is 1:
                continue

            box_result.append(box[i, :])
            cls_result.append(cls[i])
        box_result = np.array(box_result)

        return box_result, cls_result

    def del_Max_MIn(self, arr):
        max_idx = np.where(arr == np.max(arr))
        arr = np.delete(arr, max_idx[0][0])
        min_idx = np.where(arr == np.min(arr))
        arr = np.delete(arr, min_idx[0][0])

        return arr

    def cal_box_avgwh(self, bboxes):

        box_len = len(bboxes)
        box_w = np.zeros((box_len), np.int)
        box_h = np.zeros((box_len), np.int)
        i = 0
        for box in bboxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            box_w[i] = w
            box_h[i] = h
            i += 1
        box_w = self.del_Max_MIn(box_w)
        box_h = self.del_Max_MIn(box_h)

        avg_w = box_w.mean()
        avg_h = box_h.mean()

        return [avg_w, avg_h]

    def cal_dif_w(self, bboxes):

        w_dif = np.zeros([len(bboxes)], np.int)
        cx_arr = np.zeros([len(bboxes)], np.int)
        bef_cx = 0
        i = 0
        for box in bboxes:
            box_cx = (box[0] + box[2]) / 2
            cx_arr[i] = box_cx
            w_dif[i] = box_cx - bef_cx
            i += 1
            bef_cx = box_cx

        cx_res = self.del_Max_MIn(cx_arr)
        cx_avg = cx_res.mean()
        w_dif_res = self.del_Max_MIn(w_dif[1:])
        w_dif_avg = w_dif_res.mean()

        return w_dif[1:], w_dif_avg, cx_avg, cx_arr

    def cal_dif_h(self, bboxes):

        h_dif = np.zeros([len(bboxes)], np.int)
        cy_arr = np.zeros([len(bboxes)], np.int)
        bef_cy = 0
        i = 0
        for box in bboxes:
            box_cy = (box[1] + box[3]) / 2
            cy_arr[i] = box_cy
            h_dif[i] = abs(box_cy - bef_cy)
            i += 1
            bef_cy = box_cy
        cy_res = self.del_Max_MIn(cy_arr)
        cy_avg = cy_res.mean()

        h_dif_res = self.del_Max_MIn(h_dif[1:])
        h_dif_avg = h_dif_res.mean()
        return h_dif[1:], h_dif_avg, cy_avg, cy_arr

    def correct_spacing(self, bboxes, avg_wh, plate_tpAbt_lim, video_mode=False):
        box_result = bboxes
        # 去异常位置
        __, __, __, cy_arr = self.cal_dif_h(bboxes)
        w_Lres = np.where((cy_arr < plate_tpAbt_lim[0]) | (cy_arr > plate_tpAbt_lim[1]))
        arr_bbox = np.array(bboxes)
        if len(w_Lres[0]) is not 0:  # delete error rect
            w_Lres[0][0] = w_Lres[0][0]
            box_result = np.delete(arr_bbox, w_Lres[0], axis=0)
        w_dif, w_dif_avg, __, __ = self.cal_dif_w(box_result)
        w_Lres = np.where(w_dif < 0.7 * w_dif_avg)
        arr_bbox = np.array(box_result)

        if len(w_Lres[0]) is not 0:  # delete error rect
            w_Lres[0][0] = w_Lres[0][0] + 1
            box_result = np.delete(arr_bbox, w_Lres[0], axis=0)
        h_dif, h_dif_avg, cy_avg, cy_arr = self.cal_dif_h(box_result)

        h_res = np.where(h_dif >= h_dif_avg + 2) if video_mode == False else [[]]

        w_dif, w_dif_avg, __, cx_arr = self.cal_dif_w(box_result)

        w_Mres = np.where(w_dif > 1.8 * w_dif_avg)

        if len(h_res[0]) is not 0:  # fix the h
            for h_r in h_res[0]:
                box_result[h_r + 1][0] = box_result[h_r + 1][0]
                box_result[h_r + 1][1] = cy_avg - avg_wh[1] / 2
                box_result[h_r + 1][2] = box_result[h_r + 1][0] + avg_wh[0]
                box_result[h_r + 1][3] = box_result[h_r + 1][1] + avg_wh[1]

                box_result[h_r][0] = box_result[h_r][0]
                box_result[h_r][1] = cy_avg - avg_wh[1] / 2
                box_result[h_r][2] = box_result[h_r][0] + avg_wh[0]
                box_result[h_r][3] = box_result[h_r][1] + avg_wh[1]

        Cover＿idx = 2 if self.Color != '绿' else 3

        if len(w_Mres[0]) is not 0:  # Cover rect
            for w_M in w_Mres[0]:
                if w_M + 1 <= 3:  # Cover the after 2rd char
                    continue
                Cover_box_x1 = cx_arr[w_M + 1] - w_dif_avg - avg_wh[0] / 2
                Cover_box_y1 = cy_arr[w_M + 1] - avg_wh[1] / 2
                Cover_box_x2 = Cover_box_x1 + avg_wh[0]
                Cover_box_y2 = Cover_box_y1 + avg_wh[1]
                CoverBox = np.array([Cover_box_x1, Cover_box_y1, Cover_box_x2, Cover_box_y2], np.int)
                box_result = np.row_stack((box_result, CoverBox))

        if box_result[-1][0] - box_result[-2][2] > w_dif_avg * 0.3:
            box_result[-1][0] = box_result[-2][2]

        return box_result

    def unify_height(self, bboxes, avg_wh):

        bboxes[:, 3] = bboxes[:, 1] + avg_wh[1] + 4

        return bboxes

    def fix_offsetedRect(self, result, boxes, plate_img):
        Rect_Num = len(boxes)
        r_d = np.zeros(Rect_Num, np.int)
        l_d = np.zeros(Rect_Num, np.int)
        fix_result = result
        for i in range(len(boxes)):
            if i is 0:
                r_d[i] = (boxes[i + 1][0] + boxes[i + 1][2]) / 2 - (boxes[i][0] + boxes[i][2]) / 2
                l_d[i] = self.Char_avgW
            if i >= 1 and i < Rect_Num - 1:
                r_d[i] = (boxes[i + 1][0] + boxes[i + 1][2]) / 2 - (boxes[i][0] + boxes[i][2]) / 2
                l_d[i] = r_d[i - 1]
            if i is Rect_Num - 1:
                r_d[i] = self.Char_avgW
                l_d[i] = r_d[i - 1]

            # dif = abs(r_d[i] - self.Char_avgW) + abs(l_d[i] - self.Char_avgW)
            dif_l = abs(r_d[i] - self.Char_avgW)
            if dif_l >= 0.5 * self.Char_avgW and i is 2:
                boxes[i][0] = boxes[i + 1][0] - self.Char_avgW
                boxes[i][1] = boxes[i + 1][1]
                boxes[i][2] = boxes[i + 1][2] - self.Char_avgW
                boxes[i][3] = boxes[i + 1][3]

                C_char_img = plate_img[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2]]
                prob = self.char_recognition(C_char_img)
                char_fix = platechar_dic.get(str(int(prob[0])))

                fix＿reslist = list(result)
                fix＿reslist[i] = char_fix
                fix_result = ''.join(fix＿reslist)

            i += 1

        return boxes, fix_result

    def Chos_RealRect(self, bboxes, plate_tpAbt_lim):

        pass

    def Error_Correction_pre(self, bboxes, color, plate_tpAbt_lim):

        avg_wh = self.cal_box_avgwh(bboxes)
        box_res = self.correct_spacing(bboxes, avg_wh, plate_tpAbt_lim, True)
        self.Char_avgW = avg_wh[0]
        self.Char_avgH = avg_wh[1]
        # box_res = self.unify_height(box_res,avg_wh)

        return box_res

    def CoverChar_forRecog(self, index, bbox, plate_img, pro_arr, recog_result):

        K_dict = {
            '0': '-1',
            '-1': '1'
        }
        k = int(K_dict.get(str(index)))
        C_rectw = bbox[index][2] - bbox[index][0]
        C_recth = bbox[index][3] - bbox[index][1]

        C_rectx = fixEdge_W(bbox[index][0] + k * C_rectw, plate_img.shape[1])
        C_recty = fixEdge_H(bbox[index][1], plate_img.shape[0])

        CoverBox = np.array([C_rectx, C_recty, C_rectx + C_rectw, C_recty + C_recth], np.int)
        fix_bbox = np.row_stack((CoverBox, bbox)) if k is -1 else np.row_stack((bbox, CoverBox))
        # fix_bbox = np.insert(fix_bbox, 0, values=CoverBox, axis=0)
        C_char_img = plate_img[C_recty:C_recty + C_recth, C_rectx:C_rectx + C_rectw]
        prob = self.char_recognition(C_char_img)
        char_res = platechar_dic.get(str(int(prob[0])))
        pro_arr = np.row_stack((prob, pro_arr)) if k is -1 else np.row_stack((pro_arr, prob))
        fix_result = char_res + recog_result if k is -1 else recog_result + char_res

        return fix_result, pro_arr, fix_bbox

    def Error_Correction_aft(self, plate_img, bbox, recog_result, pro_arr, color):
        plateChar_Num = len(recog_result)
        fix_bbox = bbox
        fix_result = recog_result
        if color == '蓝' or color == '黄':
            if plateChar_Num is 6:

                first_char = recog_result[0]
                char_key = get_key(platechar_dic, first_char)[0]
                if int(char_key) <= 33:  # Recognize the covering ChineseChar (first)
                    fix_result, pro_arr, fix_bbox = self.CoverChar_forRecog(0, bbox, plate_img, pro_arr, recog_result)
                else:  # Recognize the covering ChineseChar (last)
                    fix_result, pro_arr, fix_bbox = self.CoverChar_forRecog(-1, bbox, plate_img, pro_arr, recog_result)

            # if plateChar_Num >= 8:
            #     Ch_char_idx = 0
            #     for i, char in enumerate(recog_result):
            #         if len(recog_result) - i + 1 >= 7:
            #             har_key = get_key(platechar_dic, char)[0]
            #             if int(har_key) > 33:
            #                 Ch_char_idx = i
            #                 continue
            #     fix_bbox = fix_bbox[Ch_char_idx:][:]
            #     pro_arr = pro_arr[Ch_char_idx:][:]
            #     fix_result = recog_result[Ch_char_idx:Ch_char_idx + 7]

        #########after output normal Plate_num:

        # fix offseted rect
        try:
            if self.layer_type == 'single':
                fix_bbox, fix_result = self.fix_offsetedRect(fix_result, fix_bbox, plate_img)
        except:
            pass

        # fix char in designated pos, fix the color and output final plate type
        last_sepCahr = ['34', '50', '63', '67']  # 澳,港,学,警
        blackchar = ['34', '50']  # 澳,港
        spe_Color = ['黑', '白']
        i = 0
        char_len = len(fix_result)
        plate_type = '未知'
        for char in fix_result:
            char_key = get_key(platechar_dic, char)[0]

            if i is 0 and not self.Color in spe_Color and int(char_key) <= 33:
                char_fix = platechar_dic.get(str(int(pro_arr[i][2])))
                fix＿reslist = list(fix_result)
                fix＿reslist[i] = char_fix
                fix_result = ''.join(fix＿reslist)

            if i is 1 and int(char_key) < 10:
                char_fix = platechar_dic.get(str(int(pro_arr[i][5])))
                fix＿reslist = list(fix_result)
                fix＿reslist[i] = char_fix
                fix_result = ''.join(fix＿reslist)

            if i is not 6 and char_key in last_sepCahr:  # Spe_char fix
                char_fix = platechar_dic.get(str(int(pro_arr[i][3])))
                fix＿reslist = list(fix_result)
                fix＿reslist[i] = char_fix
                fix_result = ''.join(fix＿reslist)

            if (i > 0 and i < 6 and int(char_key) > 33 and self.layer_type == 'single') or (
                    i is 6 and int(char_key) > 33 and not char_key in last_sepCahr):
                char_fix = platechar_dic.get(str(int(pro_arr[i][4])))
                fix＿reslist = list(fix_result)
                fix＿reslist[i] = char_fix
                fix_result = ''.join(fix＿reslist)

            # output type of license plate
            if i is 6 and int(char_key) == 36:  # plate of trailer
                fix＿reslist = list(fix_result)
                fix＿reslist[i] = '挂'
                fix_result = ''.join(fix＿reslist)
                plate_type = '挂车牌'
            if i is 6 and char_key in blackchar:  # plate of HK_MC
                self.Color = '黑'
                plate_type = '港澳牌'
            if i is 6 and int(char_key) is 70 and len(fix_result) == 8:  # plate of emergency
                fix＿reslist = list(fix_result)
                fix＿reslist[i] = '应'
                fix＿reslist[i + 1] = '急'
                fix_result = ''.join(fix＿reslist)
                plate_type = '应急牌'
            if i is 0 and int(char_key) is 71:  # plate of avaition
                fix＿reslist = list(fix_result)
                fix＿reslist[i + 1] = '航'
                fix_result = ''.join(fix＿reslist)
                plate_type = '民航牌'

            if i is 0 and self.layer_type == 'single' and fix_result[:2] == 'WJ': # plate of wj_single
                self.Color = '白'
                plate_type = '武警牌'

            if i is 2 and int(char_key) > 33 and self.layer_type == 'double':  # plate of wj_double
                fix＿reslist = list(fix_result)
                fix＿reslist[:2] = 'WJ'
                fix_result = ''.join(fix＿reslist)
                self.Color = '白'
                plate_type = '武警牌'

            if i is 6 and int(char_key) is 63:  #plate of teacher
                plate_type = '教练牌'

            if i is 6 and int(char_key) is 67:  #plate of police
                plate_type = '警牌'
                self.Color = '白'

            i += 1

        if char_len == 8 and self.layer_type == 'single' and plate_type == '未知':
            plate_type = '新能源牌'
            self.Color = '绿' if self.Color != '黄绿' else '黄绿'

        if char_len == 7 and self.Color == '蓝' and plate_type == '未知':
            plate_type = '普通蓝牌'

        if char_len == 7 and self.Color == '黄' and plate_type == '未知':
            plate_type = '普通黄牌'

        layer = '双层' if self.layer_type == 'double' else '单层'

        # if plate_type != '':
        self.plate_type = layer + '-' + plate_type + '-' + self.Color
        # print(self.plate_type)
        return fix_result, fix_bbox

    def cal_pos(self, box_priors, outputs, img_src, NUM_RESULTS, NUM_CLASSES, thr=None):

        loc = outputs[0].reshape((NUM_RESULTS, 4)) * 0.2
        conf = outputs[1].reshape((NUM_RESULTS, NUM_CLASSES))
        loc = np.array(loc, np.float16)
        conf = np.array(conf, np.float16)

        conf_exp = np.exp(conf)

        chos_idx = np.where(conf[:, 0] < conf[:, 1])

        # Post Process

        output = []
        conf_res = []
        cls_res = []

        box_priors_chos = box_priors[chos_idx[0], :]
        box_pri_chos_w = box_priors_chos[:, 2] - box_priors_chos[:, 0]
        box_pri_chos_h = box_priors_chos[:, 3] - box_priors_chos[:, 1]
        box_pri_chos_cx = box_priors_chos[:, 0] + box_priors_chos[:, 2]
        box_pri_chos_cy = box_priors_chos[:, 1] + box_priors_chos[:, 3]

        loc_chos = loc[chos_idx[0], :]
        conf_exp_chos = conf_exp[chos_idx[0], :]

        bbox_arr_cx = loc_chos[:, 0] * box_pri_chos_w[:] + box_pri_chos_cx[:]
        bbox_arr_cy = loc_chos[:, 1] * box_pri_chos_h[:] + box_pri_chos_cy[:]

        bbox_arr_w = np.exp(loc_chos[:, 2]) * box_pri_chos_w[:]
        bbox_arr_h = np.exp(loc_chos[:, 3]) * box_pri_chos_h[:]

        # valid_Idx = np.where(conf[:, 1] > 0.3)
        # start = time()
        thre = thr if thr else 0.3
        for i in range(0, len(conf_exp_chos)):
            idx = i
            x, y = conf_exp_chos[idx]
            if y / (x + y) <= thre:
                continue
            pb = box_priors_chos[idx]
            lc = loc_chos[idx]
            # var = box_var[i]

            pb_w = box_pri_chos_w[idx]
            pb_h = box_pri_chos_h[idx]
            pb_cx = box_pri_chos_cx[idx]
            pb_cy = box_pri_chos_cy[idx]

            bbox_cx = bbox_arr_cx[idx]
            bbox_cy = bbox_arr_cy[idx]
            bbox_w = bbox_arr_w[idx]
            bbox_h = bbox_arr_h[idx]

            xmin = bbox_cx - bbox_w  # * 0.5
            ymin = bbox_cy - bbox_h  # * 0.5
            xmax = bbox_cx + bbox_w  # * 0.5
            ymax = bbox_cy + bbox_h  # * 0.5

            output.append([xmin, ymin, xmax, ymax])
            conf_res.append(conf_exp_chos[idx][1])
            cls_res.append(1)

        output = np.array(output)
        try:
            output *= [img_src.shape[1], img_src.shape[0], img_src.shape[1], img_src.shape[0]]
        except:
            pass
        output = np.array(output, np.int)
        conf_res = np.array(conf_res)
        cls_res = np.array(cls_res)
        # end = time()
        # print('part2 cost:',end - start)

        return output, conf_res, cls_res

    def vehicle_Det(self, img_src):
        input_tensor = self.preprocess_vehicle_det(img_src)
        outputs = self.rknn_vehicleDet.inference(inputs=[input_tensor])  # ,data_format="nchw",data_type="float32"
        NUM_RESULTS = self.VEHICLE_NUM_RESULTS
        NUM_CLASSES = self.VEHICLE_NUM_CLASS
        priorbox = self.vehicle_priorbox
        box, conf, cls = self.cal_pos(priorbox, outputs, img_src, NUM_RESULTS, NUM_CLASSES, 0.3)
        box_fix, cls_fix = self.chooseScroe(box, conf, cls, 0.3)

        return box_fix, cls_fix

    def plate_det(self, img_src):

        input_tensor = self.preprocess_plate_det(img_src)
        # start = time()
        outputs = self.rknn_PlateDet.inference(inputs=[input_tensor])  # ,data_format="nchw",data_type="float32"
        # end = time()
        # print('plate_det inference cost:',end - start)
        NUM_RESULTS = self.PLATE_NUM_RESULTS
        NUM_CLASSES = self.PLATE_NUM_CLASS
        priorbox = self.plate_priorbox
        # start = time()
        box, conf, cls = self.cal_pos(priorbox, outputs, img_src, NUM_RESULTS, NUM_CLASSES)
        # end = time()
        box_fix, cls_fix = self.chooseScroe(box, conf, cls, 0.3)
        # print('cal_pos and chos cost:',end -start)
        return box_fix, cls_fix

    def platechar_det(self, img_src):

        input_tensor = self.preprocess_platechar_det(img_src)
        # start = time()
        outputs = self.rknn_PlateCharDet.inference(inputs=[input_tensor])  # ,data_format="nchw",data_type="float32"
        # end = time()
        # print('char_Det inference cost:',end - start)
        NUM_RESULTS = self.PLATECHAR_NUM_RESULTS
        NUM_CLASSES = self.PLATECHAR_NUM_CLASS
        priorbox = self.platechar_priorbox
        # start = time()
        box, conf, cls = self.cal_pos(priorbox, outputs, img_src, NUM_RESULTS, NUM_CLASSES)
        # end = time()
        # print('char_det calpos cost:',end -start)
        # start = time()
        # print('charbox len:',len(box))
        box_fix, cls_fix = self.chooseScroe(box, conf, cls, 0.3)
        # end = time()
        # print('char_det chosScore:',end -start)
        return box_fix, cls_fix

    def char_recognition(self, img_src):

        input_tensor = self.preprocess_platechar_recog(img_src)
        # start = time()
        outputs = self.rknn_PlateCharRecog.inference(inputs=[input_tensor])  # ,data_format="nchw",data_type="float32"
        # end = time()
        # print('char_recog cost:',end - start)
        # order = out.argsort()[::-1][:6]
        outputs = np.array(outputs).reshape((np.size(outputs)))

        output_prob = np.argsort(-outputs)

        CH_pro = output_prob[output_prob >= 33]
        En_pro = output_prob[output_prob < 33]

        output = output_prob[:6]
        output[2:4] = CH_pro[:2]
        output[4:6] = En_pro[:2]

        conf = outputs[output]
        output = np.hstack((output, conf))

        # output_prob = out['prob'][0].argmax()
        return output

    def char_recog_pro(self, box_res_sort, plate_img):
        plate_num = ''
        prob_arr = np.zeros((len(box_res_sort), 12), np.int)

        i = 0
        self.nCnChar_Count = 0
        for chars_box in box_res_sort:
            xmin = fixEdge_W(chars_box[0], plate_img.shape[1])
            ymin = fixEdge_H(chars_box[1], plate_img.shape[0])
            xmax = fixEdge_W(chars_box[2], plate_img.shape[1])
            ymax = fixEdge_H(chars_box[3], plate_img.shape[0])

            # h = (ymax - ymin)*100/plate_img.shape[0]
            # w = (xmax - xmin)*100/plate_img.shape[1]

            # print(h, w)

            char_img = plate_img[ymin:ymax, xmin:xmax]

            prob = self.char_recognition(char_img)

            prob_arr[i][:] = prob
            plate_num += platechar_dic.get(str(int(prob[0])))
            if int(prob[0]) >= 33:
                self.nCnChar_Count += 1
            i += 1
        prob_arr = np.array(prob_arr)
        return prob_arr, plate_num

    def fix_PlateRect(self, plate_box, plate_detArea):

        img_cx = plate_detArea.shape[1] / 2

        plate_cx = (plate_box[2] + plate_box[0]) / 2
        plate_w = plate_box[2] - plate_box[0]

        offset_c = abs(plate_cx - img_cx)

        offset_cal = 1 * np.exp(0.01 * offset_c)
        plate_box[0] = plate_box[0] - offset_cal if plate_cx > img_cx else plate_box[0] + offset_cal

        plate_box[2] = plate_box[0] + plate_w

        return plate_box

    def locate_plateDetArea(self, vehicle_box, image):
        # det_img_x = vehicle_box[0]
        # det_img_y = vehicle_box[1]
        # det_img_w = (vehicle_box[2] - vehicle_box[0])
        # det_img_h = (vehicle_box[3] - vehicle_box[1])
        #
        # cg_img_w = int(det_img_w)
        # cg_img_h = int(det_img_h * 1.1)
        # cv_img_x = int(det_img_x + cg_img_w / 8)
        # # cv_img_x = int(det_img_x)
        #
        # cg_img_y = int(det_img_y + cg_img_h * 1 / 2)
        # cg_img_h = int(cg_img_h / 2)
        # cg_img_w = int(cg_img_w * 6 / 8)
        # # cg_img_w = int(cg_img_w )
        # if cg_img_y >= image.shape[0]:
        #     cg_img_y = image.shape[0] - 1
        # cg_img_y_max = fixEdge_H(cg_img_y + cg_img_h,image.shape[0])
        # crop_img = image[cg_img_y:cg_img_y_max, cv_img_x:cv_img_x + cg_img_w, :]
        det_img_x = vehicle_box[0]
        det_img_y = vehicle_box[1]
        det_img_w = (vehicle_box[2] - vehicle_box[0])
        det_img_h = (vehicle_box[3] - vehicle_box[1])

        cg_img_w = int(det_img_w)
        cg_img_h = int(det_img_h * 1.0)  # int(det_img_h * 1.1)
        cv_img_x = int(det_img_x + cg_img_w / 8)
        # cv_img_x = int(det_img_x)

        cg_img_y = int(det_img_y + cg_img_h * 1 / 2)  # int(det_img_y + cg_img_h * 1 / 2)
        cg_img_h = int(cg_img_h / 2)
        cg_img_w = int(cg_img_w * 6 / 8)
        # cg_img_w = int(cg_img_w )
        if cg_img_y >= image.shape[0]:
            cg_img_y = image.shape[0] - 1
        crop_img = image[cg_img_y:cg_img_y + cg_img_h, cv_img_x:cv_img_x + cg_img_w, :]
        return crop_img

    def plateColor_Recog(self, img_calcolor):
        # cv2.imshow('color', img_calcolor)
        img_h = img_calcolor[..., 0]
        img_s = img_calcolor[..., 1]
        img_v = img_calcolor[..., 2]

        mask_color = (img_s > 43) & (img_v > 46)
        img_cal = img_h[mask_color]  # img_cal = img_h[(img_s>125)&(img_v>125)]
        mask_color[:, int(img_calcolor.shape[1] / 4):] = False
        img_cal_left = img_h[mask_color]
        
        img_black = img_h[img_v < 46]  # img_black = img_h[img_v < 100]
        img_white = img_h[(img_s < 30) & (img_v >= 46)]  # img_white = img_h[(img_s < 100)&(img_v >= 100)]
        black_ratio = len(img_black) / np.size(img_h)
        white_ratio = len(img_white) / np.size(img_h)
        # print(black_ratio)
        # print(white_ratio)

        h_hist = cv2.calcHist([img_cal], [0], None, [18], [0, 180])
        # s_hist = cv2.calcHist([img_bgr], [1], None, [256], [0, 256])
        # v_hist = cv2.calcHist([img_bgr], [2], None, [256], [0, 256])
        hist_prob = h_hist.argmax()
        # print(h_hist.max())
        result = np.array([abs(hist_prob - 2), abs(hist_prob - 7), abs(hist_prob - 12)])
        #print(result)
        hist_prob = result.argmin()

        plate_color = {
            '0': '黄',
            '1': '绿',
            '2': '蓝'
        }

        if black_ratio > 0.15 and white_ratio > 0.3:
            color = '黑'
            return color
        if (white_ratio > 0.3 and hist_prob != 2 and h_hist.max() > 500) or white_ratio > 0.5:
            color = '白'
            return color

        color = plate_color.get(str(hist_prob))

        if color == '绿':
            h_hist = cv2.calcHist([img_cal_left], [0], None, [18], [0, 180])
            hist_prob = h_hist.argmax()
            result = np.array([abs(hist_prob - 2), abs(hist_prob - 7), abs(hist_prob - 12)])
            hist_prob = result.argmin()
            color = '黄绿' if hist_prob == 0 else '绿'
        # plt.plot(h_hist)
        # plt.title('hue Histogram')
        # plt.xlabel('Bins')
        # plt.ylabel('# of Pixels')
        # plt.show()

        return color

    def cal_edge(self, img):
        img_w = img.shape[1]
        img_h = img.shape[0]
        thr_1 = img_w * 0.4
        thr_2 = img_h * 0.3
        # top
        top = 0
        for i in range(0, int(img_h / 2)):
            img_line = img[i, :]
            C_num = len(img_line[img_line > 0])

            if C_num > thr_1:
                top = i
                break
        # bot
        bot = 0
        for i in range(-img_h + 1, int(-img_h / 2)):
            img_line = img[-i, :]
            C_num = len(img_line[img_line > 0])
            if C_num > thr_1:
                bot = -i
                break
        # left
        left = 0
        for i in range(0, int(img_w / 2)):
            img_line = img[:, i]
            C_num = len(img_line[img_line > 0])
            if C_num > thr_2:
                left = i
                break
        # right
        right = 0
        for i in range(-img_w + 1, int(-img_w / 2)):
            img_line = img[:, -i]
            C_num = len(img_line[img_line > 0])
            if C_num > thr_2:
                right = -i
                break

        fix_h = bot - top + 0.1
        fix_w = right - left
        rat = fix_w / fix_h
        box_edge = np.zeros([4])
        if rat > 3 and rat < 4:
            box_edge = np.array([left, top, right, bot], np.int)

        # print(fix_w/fix_h)
        # print('1111111')
        # img_copy = img.copy()
        # cv2.rectangle(img_copy,(left,top),(right,bot),150,5)
        # cv2.imshow('fix_img',img_copy)
        # box_edge = np.array([left,top,right,bot],np.int)
        return box_edge

    def cal_plateColorEdge(self, color, img_hsv):
        box_edge = np.zeros([4], np.int)
        if color == '蓝':
            lower = np.array([100, 120, 100])
            upper = np.array([140, 250, 250])

            blue_mask = cv2.inRange(img_hsv, lower, upper)
            box_edge = self.cal_edge(blue_mask)
            # cv2.imshow('blue_mask',blue_mask)

        if color == '黄':
            lower = np.array([10, 120, 100])
            upper = np.array([50, 255, 255])

            yellow_mask = cv2.inRange(img_hsv, lower, upper)
            box_edge = self.cal_edge(yellow_mask)
            # cv2.imshow('yellow_mask',yellow_mask)

        return box_edge

    def fix_plateRect(self, plate_box, img_rgb):
        w = plate_box[2] - plate_box[0]
        h = plate_box[3] - plate_box[1]

        plate_ratio = w / h

        # print('w_h ratio: ', plate_ratio)
        if plate_ratio < 2.8:
            self.layer_type = 'double'
        else:
            self.layer_type = 'single'

        pArea_xmin = fixEdge_W(int(plate_box[0] - 0.3 * w), img_rgb.shape[1])
        pArea_ymin = fixEdge_H(int(plate_box[1] - 1 * h), img_rgb.shape[0])

        pArea_xmax = fixEdge_W(int(plate_box[2] + 0.3 * w), img_rgb.shape[1])
        pArea_ymax = fixEdge_H(int(plate_box[3] + 1 * h), img_rgb.shape[0])

        pArea = img_rgb[pArea_ymin:pArea_ymax, pArea_xmin:pArea_xmax, :]
        # start = time()
        img_hsv = cv2.cvtColor(pArea, cv2.COLOR_BGR2HSV)
        img_calcolor = img_hsv[plate_box[1] - pArea_ymin:plate_box[1] - pArea_ymin + h,
                       plate_box[0] - pArea_xmin:plate_box[0] - pArea_xmin + w, :]
        # end = time()
        # print('fix_rect:',end - start)
        try:
            color = self.plateColor_Recog(img_calcolor)
        except:
            color = ''

        box_edge = self.cal_plateColorEdge(color, img_hsv)
        # box_edge = plate_box

        box_egw = box_edge[2] - box_edge[0]
        box_egh = box_edge[3] - box_edge[1]
        if self.layer_type == 'single':
            rat_box = np.array([0.1, 1.3, 0.1, 1.3], np.float)
        else:
            rat_box = np.array([0.1, 0.4, 0.1, 0.4], np.float)
        result_box = plate_box
        # if box_egw >= 10 and box_egh >= 10 and box_edge[0] is not 0 and box_edge[1] is not 0:
        #     # rat_box[:] = 0.0
        #     box_edge += np.array([pArea_xmin, pArea_ymin, pArea_xmin, pArea_ymin])
        #     if box_edge[0] <= plate_box[0] - 0.1 * w:
        #         result_box[0] = box_edge[0]
        #         rat_box[0] = 0.0
        #     if box_edge[1] <= plate_box[1] - 0.3 * h:  # 0.3
        #         result_box[1] = box_edge[1]
        #         rat_box[1] = 1.0
        #     if box_edge[2] >= plate_box[2] + 0.1 * w:
        #         result_box[2] = box_edge[2]
        #         rat_box[2] = 0.0
        #     if box_edge[3] >= plate_box[3] + 0.3 * h:  # 0.3
        #         result_box[3] = box_edge[3]
        #         rat_box[3] = 1.0
        # box_edge = plate_box

        return color, result_box, rat_box

    def locate_charDetArea(self, plate_box, img_rgb, img_gray):

        color, box_edge, rat_box = self.fix_plateRect(plate_box, img_rgb)

        # print(box_edge)
        plate_w = (box_edge[2] - box_edge[0])
        plate_h = (box_edge[3] - box_edge[1])

        plate_xmin = fixEdge_W(int(box_edge[0] - rat_box[0] * plate_w), img_gray.shape[1])
        plate_ymin = fixEdge_H(int(box_edge[1] - rat_box[1] * plate_h), img_gray.shape[0])  # 0.3

        plate_xmax = fixEdge_W(int(box_edge[2] + rat_box[2] * plate_w), img_gray.shape[1])
        plate_ymax = fixEdge_H(int(box_edge[3] + rat_box[3] * plate_h), img_gray.shape[0])  # 0.3

        plate_ct_y = int((plate_ymax + plate_ymin) / 2)

        chatdet_img = img_gray
        expand_h = int(1.5 * plate_h)
        if plate_ct_y < expand_h:
            pad_img = np.zeros([expand_h, img_gray.shape[1]], dtype=np.uint8)
            chatdet_img = np.vstack((pad_img, img_gray))
            plate_ymin = 0
            plate_ymax += expand_h
            plate_box[1] += expand_h
            plate_box[3] += expand_h
        if plate_ct_y > img_gray.shape[0] - expand_h:
            pad_img = np.zeros([expand_h, img_gray.shape[1]], dtype=np.uint8)
            chatdet_img = np.vstack((img_gray, pad_img))
            plate_ymax += expand_h
        # cv2.imshow('chardet_img', chatdet_img)

        plate_img = chatdet_img[plate_ymin:plate_ymax, plate_xmin:plate_xmax]
        plate_rgb = img_rgb[plate_ymin:plate_ymax, plate_xmin:plate_xmax, :]
        # cv2.imshow('chardet_img', plate_img)
        plate_real_Top = plate_box[1] - plate_ymin
        plate_real_Bot = plate_box[3] - plate_ymin

        # plate_img = img_gray[plate_ymin:plate_ymax, plate_xmin:plate_xmax]
        # plate_rgb = img_rgb[plate_ymin:plate_ymax, plate_xmin:plate_xmax, :]
        #
        # plate_real_Top = plate_box[1] - plate_ymin
        # plate_real_Bot = plate_box[3] - plate_ymin
        plate_tpAbt_lim = [plate_real_Top, plate_real_Bot]

        return plate_img, plate_rgb, color, plate_tpAbt_lim

    def sort_double_layer(self, boxes):

        max_temp = boxes[:, 1].max()
        min_temp = boxes[:, 1].min()
        max_dist = max_temp - min_temp

        mask = max_temp - boxes[:, 1] > 0.5*max_dist
        a = len(boxes[mask, 1])

       # a = 3 if self.Color == '白' or self.Color == '绿' else 2
        index = np.argsort(boxes[:, 1])[:a]
        top_index = np.argsort(boxes[index][:, 0])
        box_top = boxes[index][top_index]
        boxes[index, 0] = np.arange(-1*a, 0)

        box_res = boxes[np.argsort(boxes[:, 0])]
        box_res[:a, :] = box_top

        return box_res

    def plate_recognition_novehicle(self, image):

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img_gray', img_gray)
        plate_img_res = img_gray.copy()

        plate_boxes, plate_cls = self.plate_det(img_gray)

        crop_img_copy = image.copy()
        result_plate = ['无车牌']
        # self.draw_rect(plate_boxes, crop_img_copy)
        if len(plate_boxes) < 1:
            print('miss plate')
            C_result = '无车牌'
            result_plate.append(C_result)

        print('detect plate')
        for plate_box in plate_boxes:
            # plate_box = self.fix_PlateRect(plate_box, img_gray)
            plate_img, plate＿rgb, color, plate_tpAbt_lim = self.locate_charDetArea(plate_box, image,
                                                                                   img_gray)
            plate_img_copy = plate_img.copy()
            # color = self.plateColor_Recog(plate＿rgb)
            cv2.rectangle(crop_img_copy, (plate_box[0], plate_box[1]), (plate_box[2], plate_box[3]), (255, 0, 0), 2)
            try:
                chars_boxes, chars_cls = self.platechar_det(plate_img)
                if len(chars_boxes) <= 4:
                    C_result = '无车牌'
                    result_plate.append(C_result)
                    continue
                self.draw_rect(chars_boxes, plate_img_copy)
                # objbox_sort = chars_boxes[np.lexsort(chars_boxes[:, ::-1].T)]
                # box_res = self.Error_Correction_pre(objbox_sort, color, plate_tpAbt_lim)
                # # box_res = objbox_sort
                # box_res_sort = box_res[np.lexsort(box_res[:, ::-1].T)]

                objbox_sort = chars_boxes[np.lexsort(chars_boxes[:, ::-1].T)]
                if self.layer_type == 'double':
                    objbox_sort = self.sort_double_layer(chars_boxes)
                if self.layer_type == 'single':
                    objbox_sort = self.Error_Correction_pre(objbox_sort, color, plate_tpAbt_lim)
                    objbox_sort = objbox_sort[np.lexsort(objbox_sort[:, ::-1].T)]
            except:
                print('Char_det error')
                continue
            start = time()
            # prob_arr ,plate_num= self.char_recog_threadpro(box_res_sort, plate_img)
            prob_arr, plate_num = self.char_recog_pro(objbox_sort, plate_img)
            end = time()
            print('char recog time:', end - start)
            if self.nCnChar_Count >= 4:
                C_result = '无车牌'
                result_plate.append(C_result)
                continue
            # try:
            C_result, C_bbox = self.Error_Correction_aft(plate_img, objbox_sort, plate_num, prob_arr, color)

            # except:
            # C_result = plate_num
            # C_bbox = box_res_sort

            if len(C_bbox) <= 5:
                C_result = '无车牌'
                # result_plate.append(C_result)
                # continue

            print(color)
            print(C_result)
            result_plate.append(C_result)
            # self.draw_rect(C_bbox, plate_img_copy)
            # self.draw_rect(box_res_sort, plate_img)
            cv2.imshow('crop_img_copy', plate_img_copy)
            plate_img_res = cv2.resize(plate_img, (100, 100))
            cv2.imshow('plate_img_copy', plate_img_res)
            # cv2.waitKey(0)

        return result_plate

    def vehicle_plate_recognition(self, image):
        # start = time()
        vehicle_boxes, vehicle_cls = self.vehicle_Det(image)
        # end = time()
        # print('vehicle_det cost:',end-start)
        result_plate = []
        self.plate_type = ''
        self.Color = ''
        self.layer_type = ''
        # plate_img_res = image.copy()
        # plate_img_res = cv2.resize(plate_img_res, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
        # cv2.imshow('image', plate_img_res)
        crop_img_rgb = []
        # img_copy = image.copy()
        for vehicle_box in vehicle_boxes:
            # print('detect the Vehicle')
            crop_img_rgb = self.locate_plateDetArea(vehicle_box, image)
            img_gray = cv2.cvtColor(crop_img_rgb, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('img_gray',img_gray)
            # self.draw_rect(vehicle_boxes, img_copy)
            # img_copy = cv2.resize(img_copy, (500, 500))
            # cv2.imshow('img_copy', img_copy)
            # crop_img_copy = crop_img_rgb.copy()
            # plate_img_res = img_gray.copy()
            try:
                # start = time()
                plate_boxes, plate_cls = self.plate_det(img_gray)
                # end = time()
                # print('plate_det cost',end - start)
            except:
                continue
            # crop_img_copy = crop_img_rgb.copy()

            if len(plate_boxes) < 1:
                # print('miss plate')
                C_result = '无车牌'
                result_plate.append(C_result)
                continue
            # self.draw_rect(plate_boxes, crop_img_copy)
            # print('detect plate')
            for plate_box in plate_boxes:
                # start = time()
                plate_img, plate＿rgb, color, plate_tpAbt_lim = self.locate_charDetArea(plate_box, crop_img_rgb,
                                                                                       img_gray)
                # end = time()
                # print('color and location:',end - start)
                # color = self.plateColor_Recog(plate＿rgb)
                # cv2.rectangle(crop_img_copy, (plate_box[0], plate_box[1]), (plate_box[2], plate_box[3]), (255, 0, 0), 2)
                # cv2.imshow('crop_img_copy', crop_img_copy)
                try:
                    # start = time()
                    chars_boxes, chars_cls = self.platechar_det(plate_img)

                    # end = time()
                    # print('platechar_det cost:',end - start)
                    if len(chars_boxes) <= 4:
                        C_result = '无车牌'
                        result_plate.append(C_result)
                        continue
                    # start = time()
                    objbox_sort = chars_boxes[np.lexsort(chars_boxes[:, ::-1].T)]
                    if self.layer_type == 'double':
                        objbox_sort = self.sort_double_layer(chars_boxes)
                    if self.layer_type == 'single':
                        objbox_sort = self.Error_Correction_pre(objbox_sort, color, plate_tpAbt_lim)
                        objbox_sort = objbox_sort[np.lexsort(objbox_sort[:, ::-1].T)]
                    # end =time()
                    # print('pre:',end - start)
                    # box_res_sort = objbox_sort
                except:
                    # print('Char_det error')
                    continue
                # start = time()

                prob_arr, plate_num = self.char_recog_pro(objbox_sort, plate_img)
                # end = time()
                # print('char recog time:',end - start)
                if self.nCnChar_Count >= 4:
                    C_result = '无车牌'
                    result_plate.append(C_result)
                    continue
                # try:
                # start = time()
                self.Color = color
                C_result, C_bbox = self.Error_Correction_aft(plate_img, objbox_sort, plate_num, prob_arr, color)
                # end = time()
                # print('aft:',end - start)

                # except:
                # C_result = plate_num
                # C_bbox = objbox_sort

                if len(C_bbox) <= 5:
                    C_result = '无车牌'
                    # result_plate.append(C_result)
                    # continue

                # print(self.Color)
                # print(C_result)

                result_plate.append(C_result)
                # self.draw_rect(C_bbox, plate_img)
                # self.draw_rect(objbox_sort, plate_img)

                # plate_img_res = cv2.resize(plate_img, (100, 100))
                # cv2.imshow('plate_img', plate_img_res)

        return result_plate, crop_img_rgb

    def cal_Rectpro(self, vehicle_boxes, img_src):
        # frame_copy = img_src.copy()
        # self.draw_rect(vehicle_boxes, frame_copy)
        self.rect_num = len(vehicle_boxes)
        self.rect_aftboxes = self.rect_preboxes
        self.rect_aftboxes[:, 4] -= 1

        if self.rect_num != 0:

            for vehicle_box in vehicle_boxes:
                Rect_1 = RECT(vehicle_box[0], vehicle_box[1], vehicle_box[2], vehicle_box[3])
                app_sy = True
                for j, rect_prebox in enumerate(self.rect_preboxes):
                    Rect_2 = RECT(rect_prebox[0], rect_prebox[1], rect_prebox[2], rect_prebox[3])
                    IOU, cross = self.cal_IOU(Rect_1, Rect_2)
                    # print(IOU)
                    if IOU > 0.45:
                        rect_prebox[6] += 1  # track_num
                        rect_prebox[4] = 0  # cal_iou_sy
                        rect_prebox[:4] = [vehicle_box[0], vehicle_box[1], vehicle_box[2], vehicle_box[3]]
                        self.rect_aftboxes[j][:] = rect_prebox

                        app_sy = False
                        break
                if not app_sy:
                    continue
                state = np.zeros([4], np.int)
                vehicle_box＿state = np.hstack((vehicle_box, state))
                # self.rect_aftboxes.append(vehicle_box＿state)
                self.rect_aftboxes = np.row_stack((self.rect_aftboxes, vehicle_box＿state))

            for res_box in self.rect_aftboxes:
                # print(res_box[4])
                if res_box[6] >= 5:
                    box = res_box[:4].reshape(1, 4)
                    box = list(box)
                    # self.draw_rect_track(box, frame_copy, int(res_box[6]))

        # frame_copy = cv2.resize(frame_copy, (int(frame_copy.shape[1] / 2), int(frame_copy.shape[0] / 2)))
        #
        # cv2.imshow('windows', frame_copy)  # 显示
        # if len(self.rect_preboxes) !=0:
        #     cv2.waitKey(int(1000 / int(1000)))  # 延迟
        # else:
        #     cv2.waitKey(int(1000 / int(1000)))  # 延迟
        # cv2.waitKey(1)
        return

    def init_trackoutput(self):
        self.none_plate_ct = 0
        self.ap_max = 0
        self.plate_max = []
        self.plate_result_lock = -1
        self.plate_result = []
        self.saveDif_min = 1000

        self.type_max_num = 0
        self.type_list_type = []
        self.type_max_type = ''
        return

    def output_platepro(self, image_src):
        final_result = 'Null'
        dele_Idx = np.where(self.rect_aftboxes[:, 4] <= -5)
        for idx in dele_Idx[0]:
            if self.rect_aftboxes[idx][7] >= 3 and self.rect_aftboxes[idx][5] != -1 and \
                    self.plate_max != self.last_result:
                self.last_result = self.plate_max
                print('missing plate out put:', self.plate_max)
                final_result = self.type_max_type + '-' + self.plate_max[0] if len(self.plate_max) != 0 \
                    else 'Null'
            if self.none_plate_ct >= 10 and self.rect_aftboxes[idx][7] <= 2 and self.rect_aftboxes[idx][5] != -1:
                self.plate_max = ['无车牌']
                self.none_plate_ct = 0
                # print('missing plate out put:', self.plate_max)
                final_result = self.plate_max[0] if len(self.plate_max) != 0 else 'Null'
            if self.plate_result_lock == idx:
                self.plate_result_lock = -1
            if self.plate_result_lock > idx:
                self.plate_result_lock -= 1
        self.rect_aftboxes = np.delete(self.rect_aftboxes, dele_Idx[0], axis=0)  # delete the missing vehicle

        result_plate = []
        # max_bot_rect = 0
        # max_bot_idx = 0
        for i, res_box in enumerate(self.rect_aftboxes):
            vehicle_box = [res_box[0], res_box[1], res_box[2], res_box[3]]
            track_n = res_box[6]
            rect_cy = (res_box[1] + res_box[3]) / 2

            if track_n >= 5 and vehicle_box[0] < image_src.shape[1] / 2 and vehicle_box[3] > image_src.shape[0] / 3 and \
                    self.rect_aftboxes[i][5] != -1 and self.rect_aftboxes[i][4] == 0:
                if self.rect_aftboxes[i][7] < 1 and self.none_plate_ct < 1 and vehicle_box[3] > 5*image_src.shape[0]/6:
                    continue

                if self.plate_result_lock != -1 and self.plate_result_lock != i:
                    print('plate result lock by ', i)
                    continue
                # print(self.rect_aftboxes[i][5])
                if self.rect_aftboxes[i][5] == 0:
                    self.init_trackoutput()
                if self.rect_aftboxes[i][5] == 3:  # lock the plate result buffer
                    self.plate_result_lock = i
                #start = time()
                # print('recog')
                result_plate = self.plate_recogByvehicleDeted(vehicle_box, image_src)
                #end = time()
                #print('recogtime:', end - start)
                # print(result_plate)
                # cv2.waitKey(0)

                if result_plate is not None and len(result_plate) == 1:
                    self.rect_aftboxes[i][5] += 1
                    ap_num = -1
                    if result_plate[0] != '无车牌':
                        self.plate_result.append(result_plate)
                        ap_num = self.plate_result.count(result_plate)

                        self.type_list_type.append(self.plate_type)
                        type_num = self.type_list_type.count(self.plate_type)

                        if type_num > self.type_max_num:
                            self.type_max_num = type_num
                            self.type_max_type = self.plate_type
                    else:
                        self.none_plate_ct += 1
                    if ap_num >= self.rect_aftboxes[i][7]:
                        self.rect_aftboxes[i][7] = ap_num
                        self.plate_max = result_plate
                    if (self.rect_aftboxes[i][5] == 100 or self.rect_aftboxes[i][7] >= 7 or vehicle_box[3] > \
                        image_src.shape[0] * 4 / 5) and self.plate_max != self.last_result:
                        # print(self.rect_aftboxes[i][5],self.ap_max)
                        if self.none_plate_ct >= 5 and self.rect_aftboxes[i][7] < 2 and vehicle_box[3] > \
                                image_src.shape[
                                    0] * 4 / 5:
                            self.plate_max = ['无车牌']
                            self.type_max_type = ''
                        final_result = self.type_max_type + '-' + self.plate_max[0] if len(self.plate_max) != 0 \
                            else 'Null'
                        self.rect_aftboxes[i][5] = -1
                        self.vehicle_conut += 1
                        # print('last result is ', self.last_result)
                        # print(self.plate_max, self.vehicle_conut, self.none_plate_ct)
                        if len(self.plate_max) != 0 and self.plate_max[0] != '无车牌':
                            self.last_result = self.plate_max

                        # cv2.waitKey(1000)
                        self.init_trackoutput()
                    y_dif = abs(rect_cy - image_src.shape[0] / 2)  # set save_img sy
                    if final_result == 'Null':
                        if y_dif < self.saveDif_min:
                            self.saveDif_min = y_dif
                            final_result = 'save_img'

        self.rect_preboxes = self.rect_aftboxes
        return final_result

    def plate_recogByvehicleDeted(self, recog_rect, image_src):
        # print(os.sched_getaffinity(0))
        # print('plate_recog_start')
        result_plate = []
        layer_type = []
        self.plate_type = ''
        self.Color = ''
        self.layer_type = ''
        crop_img_rgb = self.locate_plateDetArea(recog_rect, image_src)
        img_gray = cv2.cvtColor(crop_img_rgb, cv2.COLOR_BGR2GRAY)

        try:
            # start = time()
            plate_boxes, plate_cls = self.plate_det(img_gray)
            # end = time()
            # print('plate_det:',end - start)
        except:

            return result_plate
        # crop_img_copy = crop_img_rgb.copy()

        # self.draw_rect(plate_boxes, crop_img_copy)
        if len(plate_boxes) < 1:
            # print('miss plate')
            C_result = '无车牌'
            result_plate.append(C_result)
            return result_plate

        # print('detect plate')
        for plate_box in plate_boxes:
            # print(plate_box)
            try:
                plate_box = self.fix_PlateRect(plate_box, img_gray)
                plate_img, plate＿rgb, color, plate_tpAbt_lim = self.locate_charDetArea(plate_box, crop_img_rgb,
                                                                                       img_gray)
                # start = time()
                chars_boxes, chars_cls = self.platechar_det(plate_img)
                # end = time()
                # print('char_det')#,end - start)
                if len(chars_boxes) <= 4:
                    C_result = '无车牌'
                    result_plate.append(C_result)
                    continue

                objbox_sort = chars_boxes[np.lexsort(chars_boxes[:, ::-1].T)]
                if self.layer_type == 'double':
                    objbox_sort = self.sort_double_layer(chars_boxes)
                if self.layer_type == 'single':
                    objbox_sort = self.Error_Correction_pre(objbox_sort, color, plate_tpAbt_lim)
                    objbox_sort = objbox_sort[np.lexsort(objbox_sort[:, ::-1].T)]
            except:
                print('Char_det error')
                C_result = '无车牌'
                continue

            prob_arr, plate_num = self.char_recog_pro(objbox_sort, plate_img)
            # print('char recog')
            if self.nCnChar_Count >= 4:
                C_result = '无车牌'
                result_plate.append(C_result)
                continue
            self.Color = color
            C_result, C_bbox = self.Error_Correction_aft(plate_img, objbox_sort, plate_num, prob_arr, color)
            # print('aft_pro')
            if len(C_bbox) <= 5:
                C_result = '无车牌'

            # print(C_result)


            result_plate.append(C_result)
            layer_type.append(self.layer_type)
            # print(result_plate)

        if result_plate == []:
            result_plate = ['无车牌']
        if len(result_plate) > 1:
            try:
                NonePlate = result_plate.index('无车牌')
                result_plate.pop(NonePlate)
                layer_type.pop(NonePlate)
            except:
                pass

            try:
                NonePlate = layer_type.index('double')
                result_plate.pop(NonePlate)
                layer_type.pop(NonePlate)
            except:
                pass

        # print('plate_recog_end: ',result_plate)
        return result_plate

    def read_video(self, video_path):

        videoCapture = cv2.VideoCapture(video_path)
        basename = os.path.basename(video_path)
        # 获得码率及尺寸
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

        # 读帧
        success, frame = videoCapture.read()
        frame_resize = cv2.resize(frame, (1280, 720))
        cnt = 0
        drop_frame = random.randint(5, 5)

        while success:

            if cnt % drop_frame != 0:
                success, frame = videoCapture.read()  # 获取下一帧
                frame_resize = cv2.resize(frame, (1280, 720))

                cnt += 1
                continue

            self.track_pro(frame_resize)
            # save_dir = '/home/supernode/lisk_dpproject/plate_rk3399pro/Platepro_RknnModel_9_5/img_dir'
            # save_name = os.path.join(save_dir, basename + '_' + str(cnt) + '.jpg')
            # cv2.imwrite(save_name, frame_resize)
            # cv2.imshow('img', frame_resize)
            # cv2.waitKey(1)
            success, frame = videoCapture.read()  # 获取下一帧
            frame_resize = cv2.resize(frame, (1280, 720))
            drop_frame = random.randint(3, 3)
            cnt += 1

    def track_pro(self, image_src):
        start = time()
        vehicle_boxes, cls_fix = self.vehicle_Det(image_src)

        self.cal_Rectpro(vehicle_boxes, image_src)

        final_result = self.output_platepro(image_src)
        end = time()
        if final_result != 'Null' and final_result != 'save_img':
            print('cost time ', end -start)
            # cv2.waitKey(0)

        return final_result


def plate_main(img_src):
    print(img_src)
    img = img_src[0]
    cv2.imshow('img', img)
    cv2.waitKey(0)
    object_plate = plate_pro()
    print('rknn init done')
    object_plate.track_pro(img)
    print('track done')
