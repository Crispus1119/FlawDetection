import sys
class backBone():
   def get_frame(self):
        return sys._getframe(1)
   def get_model(self):
       frame=self.get_frame()
       self.action(frame)
   def action(self,frame):
       assert False,"Sorry you did not implement the method named: "+frame.f_code.co_name+" in "+self.__class__.__name__