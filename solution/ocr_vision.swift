// On-device text recognition with Apple's Vision framework (macOS only, nothing leaves the machine).
//
//   swiftc -O solution/ocr_vision.swift -o data/derived/ocr_vision
//   data/derived/ocr_vision < list_of_png_paths.txt
//
// For every image path read from stdin, prints one line per recognised text box:
//   <path>\t<text>\t<x>\t<y>\t<w>\t<h>\t<confidence>
// with the box in normalised image coordinates (origin bottom-left, as Vision reports it).

import Foundation
import Vision
import AppKit

while let line = readLine() {
    let path = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if path.isEmpty { continue }
    guard let image = NSImage(contentsOfFile: path),
          let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("\(path)\t<unreadable>\t0\t0\t0\t0\t0")
        continue
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    do {
        try handler.perform([request])
    } catch {
        continue
    }
    for obs in request.results ?? [] {
        guard let best = obs.topCandidates(1).first else { continue }
        let b = obs.boundingBox
        let text = best.string.replacingOccurrences(of: "\t", with: " ")
        print("\(path)\t\(text)\t\(b.origin.x)\t\(b.origin.y)\t\(b.size.width)\t\(b.size.height)\t\(best.confidence)")
    }
    fflush(stdout)
}
