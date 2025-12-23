     void replicate() { generation++; }

};

class MemoryStore {

private:

     sqlite3 *db;

public:

     explicit MemoryStore(const std::string &path) : db(nullptr) {

         if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) {

             std::string err = sqlite3_errmsg(db);

             sqlite3_close(db);

             db = nullptr;

             throw std::runtime_error("Failed to open database: " + err);

         }

         const char *sql = "CREATE TABLE IF NOT EXISTS dna (gen INTEGER PRIMARY KEY, phi REAL);";

         char *errMsg = nullptr;

         if (sqlite3_exec(db, sql, nullptr, nullptr, &errMsg) != SQLITE_OK) {

             std::string err = errMsg;

             sqlite3_free(errMsg);

             sqlite3_close(db);

             db = nullptr;

             throw std::runtime_error("SQL error: " + err);

         }

     }

     void save_state(int gen, double phi) {

         if (!db) return;

         sqlite3_stmt *stmt = nullptr;

         const char *sql = "INSERT OR REPLACE INTO dna(gen, phi) VALUES(?, ?);";

         if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {

             sqlite3_bind_int(stmt, 1, gen);

             sqlite3_bind_double(stmt, 2, phi);

             int rc = sqlite3_step(stmt);

             if (rc != SQLITE_DONE) {

                 std::cerr << "SQLite step error: " << sqlite3_errmsg(db) << std::endl;

             }

             sqlite3_finalize(stmt);

         } else {

             std::cerr << "SQLite prepare error: " << sqlite3_errmsg(db) << std::endl;


