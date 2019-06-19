from manimlib.imports import *
import os
import pyclbr
import numpy as np

class PlotFunctions(GraphScene):
    CONFIG = {
        "x_min" : 0,
        "x_axis_width" : 12,
        "y_axis_height" : 2,
        "x_max" : 6,
        "y_min" : 0,
        "y_max" : 2,
        "graph_origin" : 1.5*UP + 6*LEFT ,
        "function_color" : RED ,
        "axes_color" : GREEN,
        "x_labeled_nums" :range(0,12,1),
        "x_axis_label" :"Time",
        "y_labeled_nums" :range(0,2,1),
        "y_axis_label" :"Intensity",
    } 

    def construct(self):
        self.setup_axes(animate=True)
        func_graph=self.get_graph(self.func_to_graph,self.function_color)

        self.play(ShowCreation(func_graph))
        self.wait(2)

    def func_to_graph(self,x):
        return np.cos(2*np.pi*x) + 1

class CirclePlot(GraphScene):
    CONFIG = {
        "x_min" : -2,
        "x_max" : 2,
        "y_min" : -3,
        "y_max" : 3,
        "x_axis_width" : 2.7,
        "y_axis_height" : 3,
        "graph_origin" : 2.0*DOWN,
        "circle_color" : BLUE,
        "function_color" : RED,
        "axes_color" : GREEN,
        "x_labeled_nums" :range(-2,3,1),
        "y_labeled_nums" :range(-3,4,1),
    } 
    
    def construct(self):
      self.setup_axes(animate=True)
      circle1 = Circle(color=self.circle_color,radius=1)
      circle1.move_to(DOWN*2)
      func_graph1=self.get_graph(self.func_to_graph2,self.function_color)
      func_graph1a=self.get_graph(self.func_to_graph2, self.function_color)
        
      self.graph_origin = 2.0*UP
      self.x_min = -10
      self.x_max = 10
      self.x_axis_width = 14
      self.x_leftmost_tick = -10
      self.x_tick_frequency = 1
      self.x_labeled_nums = range(-10,11,5)
      self.setup_axes(animate=True)
      circle2 = Circle(color=self.circle_color,radius=1)
      circle2.move_to(UP*2)
      func_graph2=self.get_graph(self.func_to_graph2,self.function_color)

      
      self.play(ShowCreation(func_graph1)), self.add(func_graph1a)
      self.play(Transform(func_graph1, func_graph2))
      self.wait(1)
      self.play(ShowCreation(circle1)), self.play(ShowCreation(circle2))
      self.wait(1)
      self.play(Transform(func_graph1a, circle1)), self.play(Transform(func_graph1, circle2))
  
    
    def func_to_graph2(self,x):
        return np.cos(2*np.pi*x) + 1


