      con.commit()
      con.close()

    def get_embeddings(self, max_items: int | None = None) -> Tuple[np.ndarray, List[int]]:

