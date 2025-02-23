from abc import ABC, abstractmethod

class BaseClass(ABC):
    @abstractmethod
    def method(self, *args, **kwargs):
        pass

class ExampleOne(BaseClass):
    def method(self, arg_1, arg_2, arg_3_additional) -> None:
        pass

class ExampleTwo(BaseClass):
    def method(self, arg_1, arg_2, arg_4_additional, arg_5_additional):
        # Use arg_4_additional and arg_5_additional as needed
        pass


class A(ABC):
    @abstractmethod
    def foo(self, a, b, *args, **kwargs):
        pass

class B(A):
    def foo(self, a, b, c=42):
        pass