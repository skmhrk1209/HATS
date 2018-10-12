import h5py


class DigitStruct:

    def __init__(self, file):

        self.file = h5py.File(file, "r")
        self.digit_struct_name = self.file["digitStruct"]["name"]
        self.digit_struct_bbox = self.file["digitStruct"]["bbox"]

    def get_name(self, index):

        return "".join([chr(c[0]) for c in self.file[self.digit_struct_name[index][0]].value])

    def get_attr(self, attr):

        return [self.file[attr.value[i].item()].value[0][0]
                for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]

    def get_bbox(self, index):

        return {attr: self.get_attr(self.file[self.digit_struct_bbox[index].item()][attr])
                for attr in ["label", "top", "left", "height", "width"]}

    def get_digit_struct(self, index):

        def concat(dicts):
            concated = {}
            for dict in dicts:
                concated.update(dict)
            return concated

        return concat([self.get_bbox(index), {"name": self.get_name(index)}])

    def get_all_digit_structs(self):

        return [self.get_digit_struct(i) for i in range(len(self.digit_struct_name))]
