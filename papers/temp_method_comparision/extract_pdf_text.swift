import Foundation
import PDFKit

if CommandLine.arguments.count < 2 {
    fputs("usage: extract_pdf_text.swift <pdf-path>\n", stderr)
    exit(1)
}

let inputPath = CommandLine.arguments[1]
let url = URL(fileURLWithPath: inputPath)

guard let document = PDFDocument(url: url) else {
    fputs("failed to open pdf: \(inputPath)\n", stderr)
    exit(2)
}

if let text = document.string {
    print(text)
} else {
    fputs("no extractable text: \(inputPath)\n", stderr)
    exit(3)
}
