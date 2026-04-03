# VisQ Support

VisQ is an on-device photo and video retrieval app for iPhone.

This page provides support information for App Store review and end users.

## Contact and Help

For bug reports, questions, or App Review clarification, use:

- Support email: <mailto:thawakar.omkar@gmail.com>
- GitHub Issues: <https://github.com/OmkarThawakar/VisQ/issues>

## Common Questions

### Why does VisQ need photo-library access?

VisQ needs access to your photo library to index and search the photos and videos you choose to make available to the app.

### Does VisQ upload my photos?

Based on the current implementation, VisQ is intended to process media on device for indexing and retrieval rather than through a VisQ-operated backend.

### What media can I search?

VisQ is intended for photos and videos stored in your Apple Photos library that you allow the app to access and index.

### Does VisQ require an account?

Based on the current implementation, VisQ does not require you to create an account before using the core on-device retrieval features.

### How long does indexing take?

Indexing time depends on the number of photos and videos selected, device performance, available storage, and whether this is the first indexing run.

### Can I use the app offline?

VisQ is designed so core indexing and retrieval behavior runs locally on device. Normal search behavior is intended to work without a dedicated VisQ backend connection.

### How do I change which photos VisQ can access?

Open iOS Settings, find VisQ, and update the Photos permission. You can choose broader access, limited access, or remove access entirely.

### What should I do if the app shows no results?

- Confirm that Photos access is granted in iOS Settings.
- Make sure at least some photos or videos have been indexed.
- Allow the initial indexing run to finish before expecting complete search results.
- Check that the device has enough free storage for local app data.

### What should I do if indexing feels slow or appears stuck?

- Keep the app open for the initial run when possible.
- Make sure the device has enough free storage.
- Reduce the number of selected assets and try a smaller indexing batch first.
- Restart the app if the progress state does not recover after a long pause.

### How do I request deletion of app data?

You can remove locally stored VisQ app data by deleting the app from your device. This removes app-sandbox data but does not delete the original media from Apple Photos.

## Policy Links

- [Privacy Policy](./privacy.md)
- [Terms of Use](./terms.md)

_Last updated: April 3, 2026_
