pub struct Pipeline<T> {
    data: Vec<T>,
}

impl<T: Clone> Pipeline<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn map<U, F: Fn(&T) -> U>(&self, f: F) -> Pipeline<U> {
        Pipeline { data: self.data.iter().map(f).collect() }
    }

    pub fn filter<F: Fn(&T) -> bool>(&self, f: F) -> Pipeline<T> {
        Pipeline { data: self.data.iter().filter(|x| f(x)).cloned().collect() }
    }

    pub fn execute(self) -> Vec<T> {
        self.data
    }
}
