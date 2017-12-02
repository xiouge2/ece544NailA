#!/usr/bin/env python


#############################################################################
##
## Copyright (C) 2010 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##     the names of its contributors may be used to endorse or promote
##     products derived from this software without specific prior written
##     permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################


# These are only needed for Python v2 but are harmless for Python v3.
import sip
sip.setapi('QString', 2)
sip.setapi('QVariant', 2)

from PyQt4 import QtCore, QtGui
from train_rnn import raw2simplified
import json
from rdp import rdp

# import the required libraries
import numpy as np
import time
import random
import cPickle
import codecs
import collections
import os
import math
import csv
import tensorflow as tf
from six.moves import xrange
import sys

# libraries required for visualisation:
from IPython.display import SVG, display
import svgwrite # conda install -c omnia svgwrite=1.1.6
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
tf.logging.info("TensorFlow Version: %s", tf.__version__)

# import our command line tools
from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *

def encode(sess, sample_model, eval_model, model,
           input_strokes):
  strokes = to_big_strokes(input_strokes).tolist()
  strokes.insert(0, [0, 0, 1, 0, 0])
  
  del strokes[126:]
  #if strokes[6][4] != 1.0:
  #    print(strokes)
  
  seq_len = [len(input_strokes)]
  #draw_strokes(to_normal_strokes(np.array(strokes)))
  return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]

class ScribbleArea(QtGui.QWidget):
    """
      this scales the image but it's not good, too many refreshes really mess it up!!!
    """
    def __init__(self, parent=None):
        super(ScribbleArea, self).__init__(parent)

        self.setAttribute(QtCore.Qt.WA_StaticContents)
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 1
        self.myPenColor = QtCore.Qt.blue
        imageSize = QtCore.QSize(500, 500)
#       self.image = QtGui.QImage()
        self.image = QtGui.QImage(imageSize, QtGui.QImage.Format_RGB32)
        self.lastPoint = QtCore.QPoint()
        #self.position = {}
        self.stroke_index = 0
        self.position = []

        model_dir = './rnn_model'
        [self.hps_model, self.eval_hps_model, self.sample_hps_model] = load_model(model_dir)
        # construct the sketch-rnn model here:
        reset_graph()
        self.model = Model(self.hps_model)
        self.eval_model = Model(self.eval_hps_model, reuse=True)
        self.sample_model = Model(self.sample_hps_model, reuse=True)
    
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
    
        # loads the weights from checkpoint into our model
        load_checkpoint(self.sess, model_dir)

        self.z_means = np.load('z_means.npy')
        self.count = 0

    def openImage(self, fileName):
        loadedImage = QtGui.QImage()
        if not loadedImage.load(fileName):
            return False

        w = loadedImage.width()
        h = loadedImage.height()    
        self.mainWindow.resize(w, h)

#       newSize = loadedImage.size().expandedTo(self.size())
#       self.resizeImage(loadedImage, newSize)
        self.image = loadedImage
        self.modified = False
        self.update()
        return True

    def saveImage(self, fileName, fileFormat):
        visibleImage = self.image
        self.resizeImage(visibleImage, self.size())

        if visibleImage.save(fileName, fileFormat):
            self.modified = False
            return True
        else:
            return False

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def clearImage(self):
        self.image.fill(QtGui.qRgb(255, 255, 255))
        self.modified = True
        self.update()

    def mousePressEvent(self, event):
#       print "self.image.width() = %d" % self.image.width()
#       print "self.image.height() = %d" % self.image.height()
#       print "self.image.size() = %s" % self.image.size()
#       print "self.size() = %s" % self.size()
#       print "event.pos() = %s" % event.pos()
        if event.button() == QtCore.Qt.LeftButton:
            self.lastPoint = event.pos()
            self.scribbling = True
            self.stroke_index += 1
            #self.position[self.stroke_index] = []
            self.x_coord = [event.x()]
            self.y_coord = [event.y()]

    def mouseMoveEvent(self, event):
        if (event.buttons() & QtCore.Qt.LeftButton) and self.scribbling:
            self.x_coord.append(event.x())
            self.y_coord.append(event.y())
            #print 'mouseMoveEvent: x=%f, y=%f' % (event.x(), event.y())
            self.drawLineTo(event.pos())
            
            self.count += 1
            if self.count % 10 == 0:
                self.position.append([self.x_coord, self.y_coord])
                self.inference()
                del self.position[-1]



    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.scribbling:
            self.drawLineTo(event.pos())
            self.scribbling = False
            #self.position[self.stroke_index] = [self.x_coord, self.y_coord]
            self.position.append([self.x_coord, self.y_coord])
            self.dx_coord = [(t - s)*(256.0/500.0) for s, t in zip(self.x_coord, self.x_coord[1:])]
            self.dy_coord = [(t - s)*(256.0/500.0) for s, t in zip(self.y_coord, self.y_coord[1:])]
            #self.position[self.stroke_index]['y'] = 
            #print self.position
            #print self.dx_coord
            #print self.dy_coord
            self.inference()

    def inference(self):
        stroke = []
        simp = self.position
        for i in range(len(simp)):
            if i == 0:
                line = [[sx , sy] for sx, sy in zip(simp[i][0], simp[i][1])]
                line = rdp(line, 2.0)
                for j in range(len(line)-1):
                    stroke.append([line[j+1][0] - line[j][0],line[j+1][1] - line[j][1], 0.])
                stroke[-1][2] = 1.0
            else:
                line = [[sx , sy] for sx, sy in zip(simp[i][0], simp[i][1])]
                line = rdp(line, 2.0)
                stroke.append([line[0][0] - simp[i-1][0][-1], line[0][1] - simp[i-1][1][-1], 0.])
                for j in range(len(line)-1):
                    stroke.append([line[j+1][0] - line[j][0],line[j+1][1] - line[j][1], 0.])
                stroke[-1][2] = 1.0
        #print(simp)
        #print(stroke)
        sys.stdout = open(os.devnull, 'w')
        test_set = DataLoader(
            [stroke],
            batch_size=1,
            max_seq_length=125,
            random_scale_factor=0.0,
            augment_stroke_prob=0.0)
        sys.stdout = sys.__stdout__
        normalizing_scale_factor = test_set.calculate_normalizing_scale_factor()
        test_set.normalize(normalizing_scale_factor)
        stroke = test_set.strokes[0]
        #print(stroke)
        z = encode(self.sess, self.sample_model, self.eval_model, self.model, stroke)
        #print(z)
        
        diff = np.zeros(5)
        for j in range(5):
            diff[j] = np.linalg.norm(z - self.z_means[j])
        name = ['eye', 'finger', 'foot', 'hand', 'leg']
        print(name[np.argmin(diff)])

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(event.rect(), self.image)

    def resizeEvent(self, event):
#       print "resize event"
#       print "event = %s" % event
#       print "event.oldSize() = %s" % event.oldSize()
#       print "event.size() = %s" % event.size()

        self.resizeImage(self.image, event.size())

#       if self.width() > self.image.width() or self.height() > self.image.height():
#           newWidth = max(self.width() + 128, self.image.width())
#           newHeight = max(self.height() + 128, self.image.height())
#           print "newWidth = %d, newHeight = %d" % (newWidth, newHeight)
#           self.resizeImage(self.image, QtCore.QSize(newWidth, newHeight))
#           self.update()

        super(ScribbleArea, self).resizeEvent(event)

    def drawLineTo(self, endPoint):
        painter = QtGui.QPainter(self.image)
        painter.setPen(QtGui.QPen(self.myPenColor, self.myPenWidth,
            QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)
        self.modified = True

        # rad = self.myPenWidth / 2 + 2
        # self.update(QtCore.QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
        self.update()
        self.lastPoint = QtCore.QPoint(endPoint)

    def resizeImage(self, image, newSize):
        if image.size() == newSize:
            return

#       print "image.size() = %s" % repr(image.size())
#       print "newSize = %s" % newSize

# this resizes the canvas without resampling the image
        newImage = QtGui.QImage(newSize, QtGui.QImage.Format_RGB32)
        newImage.fill(QtGui.qRgb(255, 255, 255))
        painter = QtGui.QPainter(newImage)
        painter.drawImage(QtCore.QPoint(0, 0), image)


##  this resampled the image but it gets messed up with so many events...       
##      painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
##      painter.setRenderHint(QtGui.QPainter.HighQualityAntialiasing, True)
#       
#       newImage = QtGui.QImage(newSize, QtGui.QImage.Format_RGB32)
#       newImage.fill(QtGui.qRgb(255, 255, 255))
#       painter = QtGui.QPainter(newImage)
#       srcRect = QtCore.QRect(QtCore.QPoint(0,0), image.size())
#       dstRect = QtCore.QRect(QtCore.QPoint(0,0), newSize)
##      print "srcRect = %s" % srcRect
##      print "dstRect = %s" % dstRect
#       painter.drawImage(dstRect, image, srcRect)


        self.image = newImage

    def print_(self):
        printer = QtGui.QPrinter(QtGui.QPrinter.HighResolution)

        printDialog = QtGui.QPrintDialog(printer, self)
        if printDialog.exec_() == QtGui.QDialog.Accepted:
            painter = QtGui.QPainter(printer)
            rect = painter.viewport()
            size = self.image.size()
            size.scale(rect.size(), QtCore.Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.image.rect())
            painter.drawImage(0, 0, self.image)
            painter.end()

    def isModified(self):
        return self.modified

    def penColor(self):
        return self.myPenColor

    def penWidth(self):
        return self.myPenWidth


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.saveAsActs = []

        self.scribbleArea = ScribbleArea(self)
        self.scribbleArea.clearImage()
        self.scribbleArea.mainWindow = self  # maybe not using this?
        self.setCentralWidget(self.scribbleArea)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Scribble")
        self.resize(500, 500)

    def closeEvent(self, event):
        if self.maybeSave():
            event.accept()
        else:
            event.ignore()

    def open(self):
        if self.maybeSave():
            fileName = QtGui.QFileDialog.getOpenFileName(self, "Open File",
                QtCore.QDir.currentPath())
            if fileName:
                self.scribbleArea.openImage(fileName)

    def save(self):
        action = self.sender()
        fileFormat = action.data()
        self.saveFile(fileFormat)

    def penColor(self):
        newColor = QtGui.QColorDialog.getColor(self.scribbleArea.penColor())
        if newColor.isValid():
            self.scribbleArea.setPenColor(newColor)

    def penWidth(self):
        newWidth, ok = QtGui.QInputDialog.getInteger(self, "Scribble",
            "Select pen width:", self.scribbleArea.penWidth(), 1, 50, 1)
        if ok:
            self.scribbleArea.setPenWidth(newWidth)

    def about(self):
        QtGui.QMessageBox.about(self, "About Scribble",
            "<p>The <b>Scribble</b> example shows how to use "
            "QMainWindow as the base widget for an application, and how "
            "to reimplement some of QWidget's event handlers to receive "
            "the events generated for the application's widgets:</p>"
            "<p> We reimplement the mouse event handlers to facilitate "
            "drawing, the paint event handler to update the application "
            "and the resize event handler to optimize the application's "
            "appearance. In addition we reimplement the close event "
            "handler to intercept the close events before terminating "
            "the application.</p>"
            "<p> The example also demonstrates how to use QPainter to "
            "draw an image in real time, as well as to repaint "
            "widgets.</p>")

    def createActions(self):
        self.openAct = QtGui.QAction("&Open...", self, shortcut="Ctrl+O",
            triggered=self.open)

        for format in QtGui.QImageWriter.supportedImageFormats():
            format = str(format)

            text = format.upper() + "..."

            action = QtGui.QAction(text, self, triggered=self.save)
            action.setData(format)
            self.saveAsActs.append(action)

        self.printAct = QtGui.QAction("&Print...", self,
            triggered=self.scribbleArea.print_)

        self.exitAct = QtGui.QAction("E&xit", self, shortcut="Ctrl+Q",
            triggered=self.close)

        self.penColorAct = QtGui.QAction("&Pen Color...", self,
            triggered=self.penColor)

        self.penWidthAct = QtGui.QAction("Pen &Width...", self,
            triggered=self.penWidth)

        self.clearScreenAct = QtGui.QAction("&Clear Screen", self,
            shortcut="Ctrl+L", triggered=self.scribbleArea.clearImage)

        self.aboutAct = QtGui.QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QtGui.QAction("About &Qt", self,
            triggered=QtGui.qApp.aboutQt)

    def createMenus(self):
        self.saveAsMenu = QtGui.QMenu("&Save As", self)
        for action in self.saveAsActs:
            self.saveAsMenu.addAction(action)

        fileMenu = QtGui.QMenu("&File", self)
        fileMenu.addAction(self.openAct)
        fileMenu.addMenu(self.saveAsMenu)
        fileMenu.addAction(self.printAct)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAct)

        optionMenu = QtGui.QMenu("&Options", self)
        optionMenu.addAction(self.penColorAct)
        optionMenu.addAction(self.penWidthAct)
        optionMenu.addSeparator()
        optionMenu.addAction(self.clearScreenAct)

        helpMenu = QtGui.QMenu("&Help", self)
        helpMenu.addAction(self.aboutAct)
        helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(fileMenu)
        self.menuBar().addMenu(optionMenu)
        self.menuBar().addMenu(helpMenu)

    def maybeSave(self):
        if self.scribbleArea.isModified():
            ret = QtGui.QMessageBox.warning(self, "Scribble",
                "The image has been modified.\n"
                "Do you want to save your changes?",
                QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard |
                QtGui.QMessageBox.Cancel)
            if ret == QtGui.QMessageBox.Save:
                return self.saveFile('png')
            elif ret == QtGui.QMessageBox.Cancel:
                return False

        return True

    def saveFile(self, fileFormat):
        initialPath = QtCore.QDir.currentPath() + '/untitled.' + fileFormat

        fileName = QtGui.QFileDialog.getSaveFileName(self, "Save As",
            initialPath,
            "%s Files (*.%s);;All Files (*)" % (fileFormat.upper(), fileFormat))
        # coordinate = []
        # for key, x_y in self.scribbleArea.position.iteritems():
        #     x = x_y[0]
        #     y = x_y[1]
        #     if key 
        #     [[dx, dy, ] for dx, dy in zip(x,y)]
        simplified_pos = raw2simplified(self.scribbleArea.position)
        output_data = []
        for i in range(len(simplified_pos)):
            dx = [(t - s)*(256.0/500.0) for s, t in zip(simplified_pos[i][0], simplified_pos[i][0][1:])]
            dy = [(t - s)*(256.0/500.0) for s, t in zip(simplified_pos[i][1], simplified_pos[i][1][1:])]
            for j in range(len(dx)):
                if j == (len(dx)-1):
                    output_data.append([dx[j],dy[j],1])
                else:
                    output_data.append([dx[j],dy[j],0])
        with open('data.json', 'w') as fp:
            json.dump(output_data, fp, sort_keys=True, indent=4)
        if fileName:
            return self.scribbleArea.saveImage(fileName, fileFormat)
        return False


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
