import unittest
from typing import Optional, List, Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic.v1 import ValidationError, BaseModel

from mx_rag.utils.common import validate_params


class Person():
    MAX_LIMIT = 100
    name: str
    number: int = 10

    @validate_params(
        age=dict(validator=lambda x: x > 10),
        weight=dict(validator=lambda x: 90 <= x <= 150)
    )
    def __init__(self, age: int, weight: int, ranker: int = 1):
        self.age = age
        self.weight = weight
        self.ranker = ranker

    def call_back_fun(self, func, *args, **kwargs):
        func(*args, **kwargs)

    @validate_params(
        param1=dict(validator=lambda x: 0.0 < x < 1.0),
    )
    def validata_call_back_fun(self, param1: float, func, *args):
        func(*args)

    @validate_params(
        param1=dict(validator=lambda x: x < Person.MAX_LIMIT),
    )
    def validate_self_var(self, param1):
        pass


@validate_params(
    name=dict(validator=lambda x: isinstance(x, str)),
    weight=dict(validator=lambda x: 0 <= x <= 50)
)
class Animal(BaseModel):
    name: str
    master: Person
    weight: int = 10

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        pass

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        pass


@validate_params(
    param1=dict(validator=lambda x: isinstance(x, str) and len(x) >= 5),
    param2=dict(validator=lambda x: x > 0)
)
def non_class_funciton(param1, param2: int, param3=None):
    pass


@validate_params(
    param3=dict(validator=lambda x: x > 0)
)
def non_class_funciton1(param1, param2: int, param3: int = 50):
    pass


class TestValidateParams(unittest.TestCase):
    def test_class_scope(self):
        Person(18, 140, 2)
        Person(18, weight=140, ranker=-1)
        with self.assertRaises(ValueError):
            Person(18, 85, 10)

    def test_non_calss_funciton(self):
        non_class_funciton("hello", 1)
        with self.assertRaises(ValueError):
            non_class_funciton("hello", param2=-1, param3=5)
        with self.assertRaises(ValueError):
            non_class_funciton(1, param2=-1, param3=5)

    def test_call_back_function(self):
        person = Person(18, 140, 2)
        person.call_back_fun(non_class_funciton, "world!", param2=3, param3=5)
        person.call_back_fun(non_class_funciton, param1="world!", param2=3, param3=5)
        person.validata_call_back_fun(0.5, non_class_funciton, "world!", 5)
        with self.assertRaises(ValueError):
            person.validata_call_back_fun(1.1, non_class_funciton, "world!", 5)

    def test_default_parm_validation(self):
        non_class_funciton1(1, 2)

    def test_validate_self_var(self):
        person = Person(18, 140, 2)
        person.validate_self_var(80)
        with self.assertRaises(ValueError):
            person.validate_self_var(110)

    # 类继承BaseModel或者langchain的LLM等，也支持类变量的校验
    # 如果类继承BaseModel，需要设置arbitrary_types_allowed为true，否则校验的类型也要继承BaseModel
    def test_class_variable(self):
        person = Person(18, 140, 2)
        Animal(name="panda", master=person)
        with self.assertRaises(ValidationError):
            Animal(name="panda", master=123)
        Animal(name="panda", master=person, weight=0)
        with self.assertRaises(ValueError):
            Animal(name="panda", master=person, weight=-1)
