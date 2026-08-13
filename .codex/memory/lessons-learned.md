# Lessons Learned

- `DiffusionDataset.init_data()` returns a list and should be passed as a single constructor argument; expanding it with `*` breaks trainer dataset initialization.
