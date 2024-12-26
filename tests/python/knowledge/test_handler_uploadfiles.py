import unittest
from unittest.mock import Mock, patch
from mx_rag.document import LoaderMng
from mx_rag.knowledge import upload_files
from mx_rag.knowledge.handler import FileHandlerError
from mx_rag.knowledge.knowledge import KnowledgeDB

class TestUploadFiles(unittest.TestCase):
    @patch('mx_rag.document.LoaderMng')
    @patch('mx_rag.knowledge.knowledge.KnowledgeDB')
    def setUp(self, mock_knowledge, mock_loader):
        self.mock_knowledge = mock_knowledge
        self.mock_loader = mock_loader
        self.files = ['file1.txt', 'file2.txt']
        self.embed_func = lambda x: x
        self.force = False

    def test_upload_files_with_invalid_knowledge(self):
        with self.assertRaises(TypeError):
            upload_files(None, self.files, self.mock_loader, self.embed_func, self.force)

    def test_upload_files_with_invalid_loader(self):
        with self.assertRaises(TypeError):
            upload_files(self.mock_knowledge, self.files, None, self.embed_func, self.force)

    def test_upload_files_with_invalid_embed_func(self):
        with self.assertRaises(TypeError):
            upload_files(self.mock_knowledge, self.files, self.mock_loader, None, self.force)

    def test_upload_files_with_invalid_force(self):
        with self.assertRaises(TypeError):
            upload_files(self.mock_knowledge, self.files, self.mock_loader, self.embed_func, None)

    def test_upload_files_with_no_files(self):
        result = upload_files(self.mock_knowledge, [], self.mock_loader, self.embed_func, self.force)
        self.assertEqual(result, [])

    def test_upload_files_with_too_many_files(self):
        self.mock_knowledge.max_file_count = 1
        with self.assertRaises(FileHandlerError):
            upload_files(self.mock_knowledge, self.files, self.mock_loader, self.embed_func, self.force)

    def test_upload_files_with_file_check_failure(self):
        with patch('mx_rag.document.upload_files._check_file') as mock_check_file:
            mock_check_file.side_effect = Exception('File check failed')
            result = upload_files(self.mock_knowledge, self.files, self.mock_loader, self.embed_func, self.force)
            self.assertEqual(result, self.files)

    def test_upload_files_with_add_file_failure(self):
        with patch('mx_rag.document.upload_files._check_file'):
            with patch('mx_rag.document.upload_files.KnowledgeDB.add_file') as mock_add_file:
                mock_add_file.side_effect = Exception('Add file failed')
                result = upload_files(self.mock_knowledge, self.files, self.mock_loader, self.embed_func, self.force)
                self.assertEqual(result, self.files)

    def test_upload_files_success(self):
        with patch('mx_rag.document.upload_files._check_file'):
            with patch('mx_rag.document.upload_files.KnowledgeDB.add_file'):
                result = upload_files(self.mock_knowledge, self.files, self.mock_loader, self.embed_func, self.force)
                self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()