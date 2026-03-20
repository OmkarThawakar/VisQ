import Foundation
import SQLite3
import os

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

struct EmbeddingRecord: Codable {
    var id: Int64?
    var assetLocalIdentifier: String
    var assetType: Int
    var creationDate: Date?
    var embedding: Data
    var imageDescription: String
    var embeddingVersion: Int
    var indexedAt: Date

    static let databaseTableName = "embeddings"
}

enum EmbeddingStoreError: Error {
    case openDatabase(String)
    case prepareStatement(String)
    case executeStatement(String)
}

final class EmbeddingStore {
    private let storageURL: URL
    private let queue = DispatchQueue(label: "EmbeddingStore.sqlite")
    private var db: OpaquePointer?
    private var setupError: Error?

    init(storageURL: URL) {
        self.storageURL = storageURL
        do {
            try setupDatabase(at: storageURL)
        } catch {
            setupError = error
            AppLog.storage.error("Failed to set up database at \(storageURL.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }
    }

    deinit {
        if let db {
            sqlite3_close(db)
        }
    }

    private func setupDatabase(at url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)

        var database: OpaquePointer?
        guard sqlite3_open(url.path, &database) == SQLITE_OK, let database else {
            let message = database.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "Unknown SQLite open error"
            sqlite3_close(database)
            throw EmbeddingStoreError.openDatabase(message)
        }

        db = database

        let createTableSQL = """
        CREATE TABLE IF NOT EXISTS \(EmbeddingRecord.databaseTableName) (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_local_identifier TEXT NOT NULL UNIQUE,
            asset_type INTEGER NOT NULL,
            creation_date DOUBLE,
            embedding BLOB NOT NULL,
            image_description TEXT NOT NULL DEFAULT '',
            embedding_version INTEGER NOT NULL DEFAULT 1,
            indexed_at DOUBLE NOT NULL
        );
        """

        let createIndexSQL = """
        CREATE INDEX IF NOT EXISTS idx_asset_id
        ON \(EmbeddingRecord.databaseTableName) (asset_local_identifier);
        """

        try execute(createTableSQL)
        try execute(createIndexSQL)
        try ensureColumnExists(named: "image_description", definition: "TEXT NOT NULL DEFAULT ''")
        try backfillDescriptionsFromLegacyKeywordsIfNeeded()
    }

    func saveEmbedding(
        assetId: String,
        assetType: Int,
        creationDate: Date?,
        embedding: [Float],
        imageDescription: String
    ) async throws {
        let embeddingData = embedding.withUnsafeBufferPointer { Data(buffer: $0) }
        let indexedAt = Date().timeIntervalSince1970
        let creationTimestamp = creationDate?.timeIntervalSince1970

        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let sql = """
                    INSERT OR REPLACE INTO \(EmbeddingRecord.databaseTableName)
                    (asset_local_identifier, asset_type, creation_date, embedding, image_description, embedding_version, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """
                    try self.withStatement(sql) { statement in
                        sqlite3_bind_text(statement, 1, assetId, -1, SQLITE_TRANSIENT)
                        sqlite3_bind_int(statement, 2, Int32(assetType))
                        if let creationTimestamp {
                            sqlite3_bind_double(statement, 3, creationTimestamp)
                        } else {
                            sqlite3_bind_null(statement, 3)
                        }
                        let bindResult = embeddingData.withUnsafeBytes { bytes in
                            sqlite3_bind_blob(statement, 4, bytes.baseAddress, Int32(bytes.count), SQLITE_TRANSIENT)
                        }
                        guard bindResult == SQLITE_OK else {
                            throw EmbeddingStoreError.executeStatement(self.lastErrorMessage())
                        }
                        sqlite3_bind_text(statement, 5, imageDescription, -1, SQLITE_TRANSIENT)
                        sqlite3_bind_int(statement, 6, 1)
                        sqlite3_bind_double(statement, 7, indexedAt)

                        guard sqlite3_step(statement) == SQLITE_DONE else {
                            throw EmbeddingStoreError.executeStatement(self.lastErrorMessage())
                        }
                    }
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func countIndexedPhotos() throws -> Int {
        try queue.sync {
            let sql = "SELECT COUNT(*) FROM \(EmbeddingRecord.databaseTableName);"
            return try withStatement(sql) { statement in
                guard sqlite3_step(statement) == SQLITE_ROW else {
                    throw EmbeddingStoreError.executeStatement(lastErrorMessage())
                }
                return Int(sqlite3_column_int64(statement, 0))
            }
        }
    }

    func fetchAllEmbeddings() throws -> [EmbeddingRecord] {
        try queue.sync {
            let sql = """
            SELECT id, asset_local_identifier, asset_type, creation_date, embedding, image_description, embedding_version, indexed_at
            FROM \(EmbeddingRecord.databaseTableName)
            ORDER BY indexed_at DESC;
            """
            return try withStatement(sql) { statement in
                var records: [EmbeddingRecord] = []
                while sqlite3_step(statement) == SQLITE_ROW {
                    let id = sqlite3_column_int64(statement, 0)
                    let assetId = String(cString: sqlite3_column_text(statement, 1))
                    let assetType = Int(sqlite3_column_int(statement, 2))
                    let creationDate = sqlite3_column_type(statement, 3) == SQLITE_NULL ? nil : Date(timeIntervalSince1970: sqlite3_column_double(statement, 3))
                    let blobPointer = sqlite3_column_blob(statement, 4)
                    let blobLength = Int(sqlite3_column_bytes(statement, 4))
                    let embedding = blobPointer.map { Data(bytes: $0, count: blobLength) } ?? Data()
                    let imageDescription = sqlite3_column_text(statement, 5).map { String(cString: $0) } ?? ""
                    let embeddingVersion = Int(sqlite3_column_int(statement, 6))
                    let indexedAt = Date(timeIntervalSince1970: sqlite3_column_double(statement, 7))

                    records.append(
                        EmbeddingRecord(
                            id: id,
                            assetLocalIdentifier: assetId,
                            assetType: assetType,
                            creationDate: creationDate,
                            embedding: embedding,
                            imageDescription: imageDescription,
                            embeddingVersion: embeddingVersion,
                            indexedAt: indexedAt
                        )
                    )
                }
                return records
            }
        }
    }

    func isAssetIndexed(assetId: String) throws -> Bool {
        try queue.sync {
            let sql = "SELECT COUNT(*) FROM \(EmbeddingRecord.databaseTableName) WHERE asset_local_identifier = ?;"
            return try withStatement(sql) { statement in
                sqlite3_bind_text(statement, 1, assetId, -1, SQLITE_TRANSIENT)
                guard sqlite3_step(statement) == SQLITE_ROW else {
                    throw EmbeddingStoreError.executeStatement(lastErrorMessage())
                }
                return sqlite3_column_int64(statement, 0) > 0
            }
        }
    }

    func clearAllEmbeddings() throws {
        try queue.sync {
            let sql = "DELETE FROM \(EmbeddingRecord.databaseTableName);"
            try execute(sql)
        }
    }

    /// Updates only the `image_description` column for an already-indexed record.
    /// Use this during description regeneration to avoid re-embedding.
    func updateDescription(assetId: String, description: String) throws {
        try queue.sync {
            let sql = """
            UPDATE \(EmbeddingRecord.databaseTableName)
            SET image_description = ?
            WHERE asset_local_identifier = ?;
            """
            try withStatement(sql) { statement in
                sqlite3_bind_text(statement, 1, description, -1, SQLITE_TRANSIENT)
                sqlite3_bind_text(statement, 2, assetId,      -1, SQLITE_TRANSIENT)
                guard sqlite3_step(statement) == SQLITE_DONE else {
                    throw EmbeddingStoreError.executeStatement(lastErrorMessage())
                }
            }
        }
    }

    /// Returns records whose saved description is absent or too short to be
    /// a Qwen-VL description (heuristic threshold: < 80 characters).
    /// These are candidates for regeneration without re-embedding.
    func fetchRecordsWithEmptyDescriptions(minLength: Int = 80) throws -> [EmbeddingRecord] {
        try queue.sync {
            let sql = """
            SELECT id, asset_local_identifier, asset_type, creation_date, embedding,
                   image_description, embedding_version, indexed_at
            FROM \(EmbeddingRecord.databaseTableName)
            WHERE LENGTH(image_description) < \(minLength)
            ORDER BY indexed_at ASC;
            """
            return try withStatement(sql) { statement in
                var records: [EmbeddingRecord] = []
                while sqlite3_step(statement) == SQLITE_ROW {
                    let id           = sqlite3_column_int64(statement, 0)
                    let assetId      = String(cString: sqlite3_column_text(statement, 1))
                    let assetType    = Int(sqlite3_column_int(statement, 2))
                    let creationDate = sqlite3_column_type(statement, 3) == SQLITE_NULL
                        ? nil
                        : Date(timeIntervalSince1970: sqlite3_column_double(statement, 3))
                    let blobPointer  = sqlite3_column_blob(statement, 4)
                    let blobLength   = Int(sqlite3_column_bytes(statement, 4))
                    let embedding    = blobPointer.map { Data(bytes: $0, count: blobLength) } ?? Data()
                    let description  = sqlite3_column_text(statement, 5).map { String(cString: $0) } ?? ""
                    let version      = Int(sqlite3_column_int(statement, 6))
                    let indexedAt    = Date(timeIntervalSince1970: sqlite3_column_double(statement, 7))

                    records.append(EmbeddingRecord(
                        id: id,
                        assetLocalIdentifier: assetId,
                        assetType: assetType,
                        creationDate: creationDate,
                        embedding: embedding,
                        imageDescription: description,
                        embeddingVersion: version,
                        indexedAt: indexedAt
                    ))
                }
                return records
            }
        }
    }

    private func execute(_ sql: String) throws {
        try withStatement(sql) { statement in
            let result = sqlite3_step(statement)
            guard result == SQLITE_DONE else {
                throw EmbeddingStoreError.executeStatement(lastErrorMessage())
            }
        }
    }

    private func withStatement<T>(_ sql: String, execute work: (OpaquePointer) throws -> T) throws -> T {
        if let setupError {
            throw setupError
        }
        guard let db else {
            throw EmbeddingStoreError.openDatabase("Database not initialized at \(storageURL.path)")
        }

        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK, let statement else {
            throw EmbeddingStoreError.prepareStatement(lastErrorMessage())
        }

        defer {
            sqlite3_finalize(statement)
        }

        return try work(statement)
    }

    private func lastErrorMessage() -> String {
        guard let db, let cString = sqlite3_errmsg(db) else {
            return "Unknown SQLite error"
        }
        return String(cString: cString)
    }

    private func ensureColumnExists(named columnName: String, definition: String) throws {
        guard !(try hasColumn(named: columnName, in: EmbeddingRecord.databaseTableName)) else {
            return
        }

        let sql = "ALTER TABLE \(EmbeddingRecord.databaseTableName) ADD COLUMN \(columnName) \(definition);"
        try execute(sql)
    }

    private func backfillDescriptionsFromLegacyKeywordsIfNeeded() throws {
        guard try hasColumn(named: "keywords", in: EmbeddingRecord.databaseTableName),
              try hasColumn(named: "image_description", in: EmbeddingRecord.databaseTableName) else {
            return
        }

        let sql = """
        UPDATE \(EmbeddingRecord.databaseTableName)
        SET image_description = REPLACE(keywords, char(10), ', ')
        WHERE image_description = '' AND keywords != '';
        """
        try execute(sql)
    }

    private func hasColumn(named columnName: String, in tableName: String) throws -> Bool {
        let escapedTableName = tableName.replacingOccurrences(of: "'", with: "''")
        let sql = "PRAGMA table_info('\(escapedTableName)');"

        return try withStatement(sql) { statement in
            while sqlite3_step(statement) == SQLITE_ROW {
                guard let columnCString = sqlite3_column_text(statement, 1) else {
                    continue
                }

                if String(cString: columnCString) == columnName {
                    return true
                }
            }
            return false
        }
    }
}
