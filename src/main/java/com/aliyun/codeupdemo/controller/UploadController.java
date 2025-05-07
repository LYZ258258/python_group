package com.aliyun.codeupdemo.controller;

import org.apache.commons.fileupload.FileItem;
import org.apache.commons.fileupload.FileItemFactory;
import org.apache.commons.fileupload.disk.DiskFileItemFactory;
import org.apache.commons.fileupload.servlet.ServletFileUpload;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import javax.servlet.http.HttpServletRequest;
import java.io.File;
import java.util.List;

/**
 * @author Codeup
 */
@Controller
public class UploadController {

    @RequestMapping(value = "/upload", method = RequestMethod.POST)
    public String uploadFile(HttpServletRequest request) {
        boolean isMultipart = ServletFileUpload.isMultipartContent(request);

        if (isMultipart) {
            FileItemFactory factory = new DiskFileItemFactory();
            ServletFileUpload upload = new ServletFileUpload(factory);

            try {
                List items = upload.parseRequest(request);
                for (Object o : items) {
                    FileItem item = (FileItem) o;

                    if (!item.isFormField()) {
                        String fileName = item.getName();

                        String filePath = request.getServletContext().getRealPath("upload");
                        File path = new File(filePath);
                        if (!path.exists()) {
                            boolean status = path.mkdirs();
                            if (!status) {
                                throw new RuntimeException();
                            }
                        }

                        File uploadedFile = new File(path + "/" + fileName);
                        System.out.println(uploadedFile.getAbsolutePath());
                        item.write(uploadedFile);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
                return "redirect:/error.html";
            }
        }

        return "redirect:/success.html";
    }
}
